import json
import requests
from typing import Dict, Any, Tuple, Optional

from .param_utils import clean_params, map_parameters, COMMON_COMPLETION_PARAMS, CHAT_COMPLETION_PARAMS
from .image_utils import build_vision_content, process_image_data_string
from .logger import log_debug, log_info, log_error

class LlamaCppAPIClient:
    """Client for handling requests to llama-server."""
    
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
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
            response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
            log_debug(f"Response Status Code: {response.status_code}")
            
            try:
                response_json = response.json()
                log_debug(f"Response JSON preview:", response_json)
                return response_json, json.dumps(response_json, indent=2), "", response.status_code
            except json.JSONDecodeError:
                log_error(f"Failed to decode JSON from response. Status: {response.status_code}, Text: {response.text[:200]}")
                return "", response.text, "Invalid JSON response", 502
                
        except requests.exceptions.Timeout:
            log_error(f"Request timeout after {self.timeout}s")
            return "", "", "Request timeout", 408
        except requests.exceptions.ConnectionError:
            log_error(f"Connection error to {url}")
            return "", "", "Connection error", 503
        except requests.exceptions.RequestException as e:
            log_error(f"Request exception", e)
            return "", "", f"Request error: {str(e)}", 500

    def handle_completion(self, prompt: str, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /completion endpoint."""
        params = {"prompt": prompt}
        mapped_params = map_parameters(kwargs, COMMON_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)
        return self._make_request("/completion", params)

    def handle_chat_completions(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /v1/chat/completions endpoint."""
        messages = []
        
        # Parse existing messages if provided
        if kwargs.get("messages") and kwargs["messages"].strip():
            try:
                messages = json.loads(kwargs["messages"])
            except json.JSONDecodeError:
                pass
                
        # Add individual messages if provided
        if kwargs.get("system_message") and kwargs["system_message"].strip():
            messages.append({"role": "system", "content": kwargs["system_message"]})
            
        # Build user content (support for multimodal / vision)
        user_text = kwargs.get("user_message") or kwargs.get("prompt", "")
        image_data = kwargs.get("image_data")
        tensor_images = kwargs.get("images")
        
        # If image_data is still string, parse it
        if isinstance(image_data, str) and image_data.strip():
            image_data = process_image_data_string(image_data)
                
        user_content_items = build_vision_content(user_text, image_data, tensor_images)
        
        if user_content_items:
            # If we only have text and no images, send as simple string (OpenAI compat)
            if len(user_content_items) == 1 and user_content_items[0]["type"] == "text":
                messages.append({"role": "user", "content": user_content_items[0]["text"]})
            else:
                messages.append({"role": "user", "content": user_content_items})
                
        if kwargs.get("assistant_message") and kwargs["assistant_message"].strip():
            messages.append({"role": "assistant", "content": kwargs["assistant_message"]})
            
        params = {
            "messages": messages,
            "model": kwargs.get("model", "default"),
        }
        
        mapped_params = map_parameters(kwargs, CHAT_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)
        return self._make_request("/v1/chat/completions", params)

    def handle_embeddings(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /v1/embeddings endpoint."""
        input_text = kwargs.get("input_text") or kwargs.get("content") or kwargs.get("prompt", "")
        params = {
            "input": input_text,
            "model": kwargs.get("model", "default"),
            "encoding_format": kwargs.get("encoding_format", "float"),
        }
        params = clean_params(params)
        return self._make_request("/v1/embeddings", params)

    def handle_tokenize(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /tokenize endpoint."""
        content = kwargs.get("content") or kwargs.get("prompt", "")
        params = {
            "content": content,
            "add_special": kwargs.get("add_special", False),
            "parse_special": kwargs.get("parse_special", True),
            "with_pieces": kwargs.get("with_pieces", False),
        }
        return self._make_request("/tokenize", params)

    def handle_detokenize(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /detokenize endpoint."""
        tokens = kwargs.get("tokens", "[]")
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                tokens = []
        params = {"tokens": tokens}
        return self._make_request("/detokenize", params)

    def handle_apply_template(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /apply-template endpoint."""
        messages = kwargs.get("messages", "[]")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                messages = []
        params = {"messages": messages}
        return self._make_request("/apply-template", params)

    def handle_infill(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /infill endpoint."""
        params = {
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
            "temperature", "top_k", "top_p", "min_p", "seed", "stream",
            "n_predict", "stop_sequences", "repeat_penalty", "repeat_last_n"
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
        """Handle /v1/rerank endpoint."""
        query = kwargs.get("query", "")
        documents = kwargs.get("documents", "[]")
        if isinstance(documents, str):
            try:
                documents = json.loads(documents)
            except json.JSONDecodeError:
                documents = []
                
        params = {
            "model": kwargs.get("model", "default"),
            "query": query,
            "documents": documents,
            "top_n": kwargs.get("top_n", 10),
        }
        return self._make_request("/v1/rerank", params)