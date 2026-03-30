import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, FrozenSet, List, Tuple

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


class EndpointType(Enum):
    """Enum for endpoint types."""

    # --- Generation endpoints (content cleaning 適用) ---
    COMPLETION = "completion"
    CHAT_COMPLETION = "chat_completion"
    INFILL = "infill"

    # --- Utility endpoints (content cleaning 非適用) ---
    EMBEDDING = "embedding"
    TOKENIZE = "tokenize"
    DETOKENIZE = "detokenize"
    RERANKING = "reranking"
    APPLY_TEMPLATE = "apply_template"


# クラス外、EndpointType定義直後に配置
_GENERATION_ENDPOINTS: FrozenSet[EndpointType] = frozenset(
    {
        EndpointType.COMPLETION,
        EndpointType.CHAT_COMPLETION,
        EndpointType.INFILL,
    }
)


@dataclass
class ApiResponse:
    """Typed container for llama-server responses."""

    data: Dict[str, Any]
    raw: str = ""
    error: str = ""
    status_code: int = 200
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def unpack(self) -> Tuple[Dict[str, Any], str, str, int, List[Dict[str, Any]]]:
        """Unpack response data into components for node processing.

        Returns:
            Tuple of (data, raw, error, status_code, metadata)
        """
        return self.data, self.raw, self.error, self.status_code, self.metadata


class LlamaCppClientError(Exception):
    """Base exception for LlamaCppClient errors."""

    pass


class LlamaConnectionError(LlamaCppClientError):
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
                        raise LlamaConnectionError(
                            f"Connection failed after {max_retries} retries: {str(e)}"
                        ) from e
            return None

        return wrapper

    return decorator


class LlamaCppAPIClient:
    """Client for handling requests to llama-server."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = None
        self._request_start_time = None

    def _get_session(self) -> requests.Session:
        """Get or create a requests session."""
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _close_session(self) -> None:
        """Close the session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _reset_timing(self) -> None:
        self._request_start_time = None

    def _clean_content(self, content_str: str) -> str:
        """Extract content: prioritize <prompt> tag, then remove <think> tags."""
        if not content_str:
            return ""

        # Try to extract <prompt> tag content first
        match = re.search(r"<prompt>(.*?)</prompt>", content_str, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Otherwise remove <think> tags
        cleaned = re.sub(r"<think>.*?</think>", "", content_str, flags=re.DOTALL).strip()
        return cleaned

    def _validate_response_structure(self, response_json: Any, endpoint_type: EndpointType) -> bool:
        """Validate response JSON structure based on endpoint type.

        Args:
            response_json: Parsed JSON response (dict or list)
            endpoint_type: The endpoint type to validate against

        Returns:
            True if the response structure is valid for the given endpoint type
        """
        if not isinstance(response_json, dict):
            return False

        if endpoint_type == EndpointType.CHAT_COMPLETION:
            return bool(response_json.get("choices"))
        elif endpoint_type in {
            EndpointType.COMPLETION,
            EndpointType.INFILL,
            EndpointType.DETOKENIZE,
            EndpointType.APPLY_TEMPLATE,
        }:
            return "content" in response_json or "text" in response_json
        elif endpoint_type == EndpointType.EMBEDDING:
            return "data" in response_json
        elif endpoint_type == EndpointType.TOKENIZE:
            return "tokens" in response_json
        elif endpoint_type == EndpointType.RERANKING:
            return "results" in response_json
        else:
            return True  # Unknown endpoint: フォールバック（寛容に）

    def _apply_content_cleaning(
        self, response_json: Dict[str, Any], endpoint_type: EndpointType
    ) -> None:
        """Apply content cleaning (remove <think> tags, extract <prompt>) in-place.

        Only called for generation endpoints (COMPLETION, CHAT_COMPLETION, INFILL).
        Mutates response_json in-place.

        Args:
            response_json: Response dictionary to modify in-place
            endpoint_type: The endpoint type being processed
        """
        content_str = ""
        if endpoint_type == EndpointType.CHAT_COMPLETION:
            choices = response_json.get("choices", [])
            if choices:
                content_str = choices[0].get("message", {}).get("content", "")
        elif "content" in response_json:
            content_str = response_json["content"]
        elif "text" in response_json:
            content_str = response_json["text"]

        if content_str:
            cleaned_content = self._clean_content(content_str)
            if endpoint_type == EndpointType.CHAT_COMPLETION:
                response_json["choices"][0]["message"]["content"] = cleaned_content
            elif "content" in response_json:
                response_json["content"] = cleaned_content
            elif "text" in response_json:
                response_json["text"] = cleaned_content

    def _parse_and_validate(
        self, raw_response: requests.Response, endpoint_type: EndpointType
    ) -> ApiResponse:
        """Parse response JSON, validate structure, and clean content.

        Content cleaning (_clean_content) is only applied to generation endpoints
        (COMPLETION, CHAT_COMPLETION, INFILL) to remove <think> tags and extract
        <prompt> content. Utility endpoints return data as-is.

        Args:
            raw_response: Raw HTTP response object
            endpoint_type: The endpoint type to validate against

        Returns:
            ApiResponse with parsed data or error message
        """
        # Phase 1: HTTP status check
        if raw_response.status_code >= 400:
            error_text = raw_response.text[:200] if raw_response.text else ""
            error_msg = f"HTTP {raw_response.status_code}: {error_text}"
            return ApiResponse(
                data={},
                error=error_msg,
                status_code=raw_response.status_code,
            )

        # Phase 2: JSON parse
        try:
            response_json = raw_response.json()
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in response: {str(e)}"
            return ApiResponse(
                data={},
                error=error_msg,
                status_code=raw_response.status_code,
            )

        # Phase 3: エンドポイント別バリデーション
        is_valid = self._validate_response_structure(response_json, endpoint_type)

        if not is_valid:
            error_msg = f"Invalid response structure for {endpoint_type.value}"
            return ApiResponse(
                data={},
                error=error_msg,
                status_code=raw_response.status_code,
            )

        # Phase 4: 生成系エンドポイントのみ content cleaning
        if endpoint_type in _GENERATION_ENDPOINTS:
            self._apply_content_cleaning(response_json, endpoint_type)

        return ApiResponse(
            data=response_json,
            raw=json.dumps(response_json, indent=2),
            error="",
            status_code=raw_response.status_code,
        )

    @retry_on_transient_error(max_retries=3, base_delay=1.0)
    def _make_request(
        self,
        endpoint_path: str,
        data: Dict[str, Any],
        endpoint_type: EndpointType,
    ) -> ApiResponse:
        """Make HTTP POST request to llama-server and return ApiResponse."""
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
            log_debug(f"Request completed in {elapsed_time:.2f}s")  # noqa: E231

            return self._parse_and_validate(response, endpoint_type)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            raise
        except (RequestError, ResponseError):
            raise
        except Exception as e:
            log_error(f"Unexpected error in _make_request: {e}", e)
            return ApiResponse(
                data={},
                error=str(e),
                status_code=500,
            )

    @retry_on_transient_error(max_retries=2, base_delay=0.5)
    def get_health(self) -> Dict[str, Any]:
        """GET /health - Check server status and slot availability."""
        url = f"{self.base_url}/health"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        log_debug(f"Checking health: {url}")
        try:
            session = self._get_session()
            response = session.get(url, headers=headers, timeout=5)
            return response.json() if response.status_code == 200 else {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @retry_on_transient_error(max_retries=2, base_delay=0.5)
    def get_props(self) -> Dict[str, Any]:
        """GET /props - Get server and model metadata."""
        url = f"{self.base_url}/props"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        log_debug(f"Getting props: {url}")
        try:
            session = self._get_session()
            response = session.get(url, headers=headers, timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}

    def is_moe_model(self) -> bool:
        """Detect if the loaded model uses a MoE architecture.

        Strategy (applied in order):
        1. Structural check: look for 'expert_count' key in the props dict tree.
           This is the most reliable signal, as it appears in DeepSeek and similar
           models that explicitly expose their expert configuration.
        2. Keyword check: scan the JSON-serialised props string for architecture
           names associated with MoE layouts (mixtral, deepseek, moe, experts).

        Returns:
            True if either heuristic matches, False otherwise.
        """
        props = self.get_props()

        # --- 1. Structural check for explicit expert_count field ---
        def _has_expert_count(obj: Any, depth: int = 0) -> bool:
            """Recursively search for 'expert_count' key."""
            if depth > 5:
                return False
            if isinstance(obj, dict):
                if "expert_count" in obj:
                    return True
                return any(_has_expert_count(v, depth + 1) for v in obj.values())
            if isinstance(obj, list):
                return any(_has_expert_count(item, depth + 1) for item in obj)
            return False

        if _has_expert_count(props):
            log_info("MoE Model detected via structural 'expert_count' field.")
            return True

        # --- 2. Keyword check on serialised props string ---
        props_str = json.dumps(props).lower()
        moe_keywords = ["mixtral", "deepseek", "moe", "experts"]
        for kw in moe_keywords:
            if kw in props_str:
                log_info(f"MoE Model detected via keyword: '{kw}'")
                return True

        return False


    def handle_completion(self, prompt: str, **kwargs) -> ApiResponse:
        """Handle /completion endpoint."""
        params: Dict[str, Any] = {"prompt": prompt}
        mapped_params = map_parameters(kwargs, COMMON_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)

        # 5090 God Speed Mode Stop sequence
        params.update({"stop": ["</prompt>", "</s>", "\n\n"]})

        return self._make_request("/completion", params, EndpointType.COMPLETION)

    def _add_system_message(self, messages: List[Dict[str, Any]], system_message: str) -> None:
        """Add system message to the beginning of message list."""
        if isinstance(system_message, str) and system_message.strip():
            messages.insert(0, {"role": "system", "content": system_message})

    def _add_message_history(self, messages: List[Dict[str, Any]], messages_input: Any) -> None:
        """Add message history to message list."""
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

    def _add_user_content(self, messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Add user content (text + images) to message list and return metadata.

        Returns:
            List of metadata dictionaries
        """
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

        return metadata_list

    def _add_assistant_prefill(
        self, messages: List[Dict[str, Any]], assistant_message: str
    ) -> None:
        """Add assistant message (prefill) to message list."""
        if assistant_message and assistant_message.strip():
            messages.append({"role": "assistant", "content": assistant_message})

    def _build_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build ordered message list: system → history → user → assistant.

        Returns:
            Tuple of (messages_list, metadata_list)
        """
        messages: List[Dict[str, Any]] = []

        # 1. Add system message first
        self._add_system_message(messages, kwargs.get("system_message", ""))

        # 2. Add message history
        self._add_message_history(messages, kwargs.get("messages"))

        # 3. Add user content (text + images, returns metadata)
        user_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
        metadata_list = self._add_user_content(messages, **user_kwargs)

        # 4. Add assistant prefill
        self._add_assistant_prefill(messages, kwargs.get("assistant_message", ""))

        return messages, metadata_list

    def handle_chat_completions(self, **kwargs) -> ApiResponse:
        """Handle /v1/chat/completions endpoint."""
        messages, metadata_list = self._build_messages(**kwargs)

        params: Dict[str, Any] = {"messages": messages, "model": kwargs.get("model", "default")}
        params.update(map_parameters(kwargs, CHAT_COMPLETION_PARAMS))
        params = clean_params(params)

        api_response = self._make_request(
            "/v1/chat/completions", params, EndpointType.CHAT_COMPLETION
        )
        api_response.metadata = metadata_list
        return api_response

    def handle_embeddings(self, **kwargs) -> ApiResponse:
        """Handle /v1/embeddings endpoint."""
        params: Dict[str, Any] = {}
        params["input"] = kwargs.get("input_text", "")
        if model := kwargs.get("model"):
            params["model"] = model
        if encoding_format := kwargs.get("encoding_format"):
            params["encoding_format"] = encoding_format
        params = clean_params(params)

        return self._make_request("/v1/embeddings", params, EndpointType.EMBEDDING)

    def handle_tokenize(self, **kwargs) -> ApiResponse:
        """Handle /tokenize endpoint."""
        params: Dict[str, Any] = {}
        params["content"] = kwargs.get("content", "")
        if kwargs.get("add_special") is not None:
            params["add_special"] = kwargs.get("add_special")
        if kwargs.get("parse_special") is not None:
            params["parse_special"] = kwargs.get("parse_special")
        if kwargs.get("with_pieces") is not None:
            params["with_pieces"] = kwargs.get("with_pieces")
        params = clean_params(params)

        return self._make_request("/tokenize", params, EndpointType.TOKENIZE)

    def handle_detokenize(self, **kwargs) -> ApiResponse:
        """Handle /detokenize endpoint."""
        params: Dict[str, Any] = {}
        tokens = kwargs.get("tokens", [])
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                tokens = []
        params["tokens"] = tokens
        params = clean_params(params)

        return self._make_request("/detokenize", params, EndpointType.DETOKENIZE)

    def handle_apply_template(self, **kwargs) -> ApiResponse:
        """Handle /apply_template endpoint."""
        params: Dict[str, Any] = {}
        messages = kwargs.get("messages", [])
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                messages = []
        params["messages"] = messages
        params = clean_params(params)

        return self._make_request("/apply_template", params, EndpointType.APPLY_TEMPLATE)

    def handle_infill(self, **kwargs) -> ApiResponse:
        """Handle /infill endpoint."""
        params: Dict[str, Any] = {
            "input_prefix": kwargs.get("input_prefix", ""),
            "input_suffix": kwargs.get("input_suffix", ""),
            "n_predict": kwargs.get("n_predict", -1),
        }
        mapped_params = map_parameters(kwargs, COMMON_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)

        return self._make_request("/infill", params, EndpointType.INFILL)

    def handle_reranking(self, **kwargs) -> ApiResponse:
        """Handle /reranking endpoint."""
        params: Dict[str, Any] = {}
        if model := kwargs.get("model"):
            params["model"] = model
        params["query"] = kwargs.get("query", "")
        documents = kwargs.get("documents", [])
        if isinstance(documents, str):
            try:
                documents = json.loads(documents)
            except json.JSONDecodeError:
                documents = []
        params["documents"] = documents
        if top_n := kwargs.get("top_n"):
            params["top_n"] = top_n
        params = clean_params(params)

        return self._make_request("/reranking", params, EndpointType.RERANKING)

    def close(self) -> None:
        """Close the session and reset timing."""
        self._close_session()
        self._reset_timing()
