import json
from typing import Any, Dict, Optional, Tuple, TypedDict

try:
    import torch
except ImportError:
    torch = None

try:
    from utils.llama_client import ApiResponse, LlamaCppAPIClient
    from utils.logger import log_error, log_info, set_debug_mode
    from utils.param_utils import parse_json_param
except ImportError:
    from .utils.llama_client import ApiResponse, LlamaCppAPIClient  # type: ignore[no-redef]
    from .utils.logger import log_error, log_info, set_debug_mode  # type: ignore[no-redef]
    from .utils.param_utils import parse_json_param  # type: ignore[no-redef]


class SamplingParams(TypedDict):
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    seed: int
    repeat_penalty: float
    repeat_last_n: int
    presence_penalty: float
    frequency_penalty: float
    dry_multiplier: float
    dry_base: float
    dry_allowed_length: int
    dry_penalty_last_n: int
    dry_sequence_breakers: str
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    typical_p: float
    n_keep: int
    stop_sequences: str
    ignore_eos: bool
    stream: bool
    n_probs: int
    min_keep: int
    post_sampling_probs: bool
    return_tokens: bool
    timings_per_token: bool
    grammar: str
    logit_bias: str
    cache_prompt: bool
    id_slot: int
    samplers: str
    t_max_predict_ms: int
    lora: str
    dynatemp_range: float
    dynatemp_exponent: float
    xtc_probability: float
    xtc_threshold: float


class CompletionParams(TypedDict):
    prompt: str
    n_predict: int
    sampling: SamplingParams


class InfillParams(TypedDict):
    input_prefix: str
    input_suffix: str
    input_extra: str
    prompt: str
    n_predict: int
    sampling: SamplingParams


class ChatParams(TypedDict):
    messages: str
    system_message: str
    user_message: str
    prompt: str
    assistant_message: str
    max_tokens: int
    model: str
    tools: str
    tool_choice: str
    response_format: str
    image_data: str
    images: Optional[torch.Tensor]
    extract_metadata: bool
    sampling: SamplingParams


class RerankingParams(TypedDict):
    model: str
    query: str
    documents: str
    top_n: int


class NodeResponse(TypedDict):
    response: str
    raw_response: str
    error: str
    status_code: int
    metadata: Dict[str, Any]


class LlamaCppClientNode:
    """
    ComfyUI custom node that acts as a client for llama-server from llama.cpp.
    Supports ALL possible parameters that llama-server accepts through its various endpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional: Dict[str, Any] = {}
        optional.update(cls._endpoint_specific_input_types())
        optional.update(cls._sampling_input_types())
        optional.update(cls._dry_input_types())
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080",
                        "multiline": False,
                        "tooltip": "Base URL of the llama-server instance",
                    },
                ),
                "endpoint": (
                    [
                        "completion",
                        "chat_completions",
                        "embeddings",
                        "tokenize",
                        "detokenize",
                        "apply_template",
                        "infill",
                        "reranking",
                    ],
                    {"default": "completion", "tooltip": "API endpoint to use"},
                ),
            },
            "optional": optional,
        }

    @classmethod
    def _sampling_input_types(cls) -> Dict[str, Any]:
        return {
            "temperature": (
                "FLOAT",
                {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Sampling temperature",
                },
            ),
            "top_k": (
                "INT",
                {"default": 40, "min": 0, "max": 1000, "tooltip": "Top-k sampling"},
            ),
            "top_p": (
                "FLOAT",
                {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-p sampling",
                },
            ),
            "min_p": (
                "FLOAT",
                {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Min-p sampling",
                },
            ),
            "seed": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed (-1 for random)",
                },
            ),
            "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 5.0, "step": 0.01}),
            "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 2048}),
            "presence_penalty": (
                "FLOAT",
                {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
            ),
            "frequency_penalty": (
                "FLOAT",
                {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
            ),
            "mirostat": ("INT", {"default": 0, "min": 0, "max": 2}),
            "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "mirostat_eta": (
                "FLOAT",
                {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001},
            ),
            "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "dynatemp_range": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            "dynatemp_exponent": (
                "FLOAT",
                {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01},
            ),
            "xtc_probability": (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "xtc_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "n_keep": ("INT", {"default": 0, "min": -1, "max": 2048}),
            "stop_sequences": ("STRING", {"default": "[]", "multiline": True}),
            "ignore_eos": ("BOOLEAN", {"default": False}),
            "stream": ("BOOLEAN", {"default": False}),
            "n_probs": ("INT", {"default": 0, "min": 0, "max": 100}),
            "min_keep": ("INT", {"default": 0, "min": 0, "max": 100}),
            "post_sampling_probs": ("BOOLEAN", {"default": False}),
            "return_tokens": ("BOOLEAN", {"default": False}),
            "timings_per_token": ("BOOLEAN", {"default": False}),
            "grammar": ("STRING", {"default": "", "multiline": True}),
            "logit_bias": ("STRING", {"default": "[]", "multiline": True}),
            "cache_prompt": ("BOOLEAN", {"default": True}),
            "id_slot": ("INT", {"default": -1, "min": -1, "max": 100}),
            "samplers": (
                "STRING",
                {
                    "default": '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]',
                    "multiline": False,
                },
            ),
            "t_max_predict_ms": ("INT", {"default": 0, "min": 0, "max": 60000}),
            "lora": ("STRING", {"default": "[]", "multiline": True}),
        }

    @classmethod
    def _dry_input_types(cls) -> Dict[str, Any]:
        return {
            "dry_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            "dry_base": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 5.0, "step": 0.01}),
            "dry_allowed_length": ("INT", {"default": 2, "min": 1, "max": 100}),
            "dry_penalty_last_n": ("INT", {"default": -1, "min": -1, "max": 2048}),
            "dry_sequence_breakers": (
                "STRING",
                {"default": '["\\n", ":", "\\"", "*"]', "multiline": False},
            ),
        }

    @classmethod
    def _endpoint_specific_input_types(cls) -> Dict[str, Any]:
        return {
            "prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The prompt text for completion/chat",
                },
            ),
            "system_message": (
                "STRING",
                {"default": "", "multiline": True, "tooltip": "System message for chat"},
            ),
            "user_message": (
                "STRING",
                {"default": "", "multiline": True, "tooltip": "User message for chat"},
            ),
            "n_predict": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 1000000,
                    "tooltip": "Number of tokens to predict (-1 = infinity)",
                },
            ),
            "json_schema": ("STRING", {"default": "", "multiline": True}),
            "response_fields": ("STRING", {"default": "[]", "multiline": False}),
            "assistant_message": ("STRING", {"default": "", "multiline": True}),
            "messages": ("STRING", {"default": "[]", "multiline": True}),
            "max_tokens": ("INT", {"default": -1, "min": -1, "max": 1000000}),
            "model": ("STRING", {"default": "", "multiline": False}),
            "tools": ("STRING", {"default": "[]", "multiline": True}),
            "tool_choice": ("STRING", {"default": "auto", "multiline": False}),
            "response_format": ("STRING", {"default": "", "multiline": True}),
            "input_text": ("STRING", {"default": "", "multiline": True}),
            "encoding_format": (["float", "base64"], {"default": "float"}),
            "embd_normalize": ("INT", {"default": 2, "min": -1, "max": 10}),
            "content": ("STRING", {"default": "", "multiline": True}),
            "tokens": ("STRING", {"default": "[]", "multiline": False}),
            "add_special": ("BOOLEAN", {"default": False}),
            "parse_special": ("BOOLEAN", {"default": True}),
            "with_pieces": ("BOOLEAN", {"default": False}),
            "input_prefix": ("STRING", {"default": "", "multiline": True}),
            "input_suffix": ("STRING", {"default": "", "multiline": True}),
            "input_extra": ("STRING", {"default": "[]", "multiline": True}),
            "query": ("STRING", {"default": "", "multiline": True}),
            "documents": ("STRING", {"default": "[]", "multiline": True}),
            "top_n": ("INT", {"default": 10, "min": 1, "max": 1000}),
            "api_key": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API key for authentication (if required)",
                },
            ),
            "timeout": (
                "INT",
                {
                    "default": 600,
                    "min": 1,
                    "max": 3600,
                    "tooltip": "Request timeout in seconds",
                },
            ),
            "image_data": ("STRING", {"default": "[]", "multiline": True}),
            "images": (
                "IMAGE",
                {"tooltip": "ComfyUI image tensor to send to the multimodal model"},
            ),
            "moe_mode": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Enable MoE optimization (disable heavy sampling stats)",
                },
            ),
            "extract_metadata": ("BOOLEAN", {"default": True}),
            "debug_mode": ("BOOLEAN", {"default": True}),
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "JSON")
    RETURN_NAMES = ("response", "raw_response", "error", "status_code", "metadata")
    FUNCTION = "process_request"
    CATEGORY = "AI/LlamaCpp"

    @staticmethod
    def _extract_response_text(response: Dict[str, Any], endpoint: str) -> str:
        """Extract response text from an API response payload.

        Handles all endpoint types:
        - chat_completions: Extract from choices[0].message.content
        - embeddings: Convert data array to JSON string
        - tokenize: Convert tokens array to JSON string
        - reranking: Convert results to JSON string
        - completion, infill, detokenize, apply_template: Extract content or text field
        """
        if isinstance(response, dict):
            if endpoint == "chat_completions":
                choices = response.get("choices", [])
                if choices and len(choices) > 0:
                    return choices[0].get("message", {}).get("content", "")
                return ""
            elif endpoint == "embeddings":
                return json.dumps(response.get("data", []))
            elif endpoint == "tokenize":
                return json.dumps(response.get("tokens", []))
            elif endpoint == "reranking":
                return json.dumps(response.get("results", []))
            else:
                return str(response.get("content", response.get("text", "")))
        return str(response) if response else ""

    @staticmethod
    def _build_node_response(
        response_text: str,
        raw_response: str,
        error: str,
        status_code: int,
        metadata: Dict[str, Any],
    ) -> NodeResponse:
        return {
            "response": response_text,
            "raw_response": raw_response,
            "error": error,
            "status_code": status_code,
            "metadata": metadata,
        }

    def _build_sampling_kwargs(self, params: SamplingParams) -> Dict[str, Any]:
        return {
            "temperature": params["temperature"],
            "top_k": params["top_k"],
            "top_p": params["top_p"],
            "min_p": params["min_p"],
            "seed": params["seed"],
            "repeat_penalty": params["repeat_penalty"],
            "repeat_last_n": params["repeat_last_n"],
            "presence_penalty": params["presence_penalty"],
            "frequency_penalty": params["frequency_penalty"],
            "dry_multiplier": params["dry_multiplier"],
            "dry_base": params["dry_base"],
            "dry_allowed_length": params["dry_allowed_length"],
            "dry_penalty_last_n": params["dry_penalty_last_n"],
            "dry_sequence_breakers": parse_json_param(params.get("dry_sequence_breakers", ""), []),
            "stop": parse_json_param(params["stop_sequences"], []),
            "stream": params["stream"],
            "cache_prompt": params["cache_prompt"],
            "id_slot": params["id_slot"],
            "samplers": parse_json_param(params["samplers"], []),
            "t_max_predict_ms": params["t_max_predict_ms"],
            "grammar": params["grammar"],
            "logit_bias": parse_json_param(params["logit_bias"], []),
            "n_probs": params["n_probs"],
            "min_keep": params["min_keep"],
            "post_sampling_probs": params["post_sampling_probs"],
            "return_tokens": params["return_tokens"],
            "timings_per_token": params["timings_per_token"],
            "ignore_eos": params["ignore_eos"],
            "n_keep": params["n_keep"],
            "dynatemp_range": params["dynatemp_range"],
            "dynatemp_exponent": params["dynatemp_exponent"],
            "xtc_probability": params["xtc_probability"],
            "xtc_threshold": params["xtc_threshold"],
            "mirostat": params["mirostat"],
            "mirostat_tau": params["mirostat_tau"],
            "mirostat_eta": params["mirostat_eta"],
            "typical_p": params["typical_p"],
            "lora": parse_json_param(params["lora"], []),
        }

    def _build_completion_kwargs(self, params: CompletionParams) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "prompt": params["prompt"],
            "n_predict": params["n_predict"],
        }
        kwargs.update(self._build_sampling_kwargs(params["sampling"]))
        return kwargs

    def _build_infill_kwargs(self, params: InfillParams) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "input_prefix": params["input_prefix"],
            "input_suffix": params["input_suffix"],
            "input_extra": parse_json_param(params["input_extra"], []),
            "prompt": params["prompt"],
            "n_predict": params["n_predict"],
        }
        kwargs.update(self._build_sampling_kwargs(params["sampling"]))
        return kwargs

    def _build_chat_kwargs(self, params: ChatParams) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "messages": params["messages"],
            "system_message": params["system_message"],
            "user_message": params["user_message"],
            "prompt": params["prompt"],
            "assistant_message": params["assistant_message"],
            "max_tokens": params["max_tokens"],
            "model": params["model"],
            "tools": parse_json_param(params["tools"], []),
            "tool_choice": params["tool_choice"],
            "response_format": parse_json_param(params["response_format"], None),
            "image_data": params["image_data"],
            "images": params["images"],
            "extract_metadata": params["extract_metadata"],
        }
        kwargs.update(self._build_sampling_kwargs(params["sampling"]))
        return kwargs

    def process_request(
        self,
        server_url: str,
        endpoint: str,
        prompt: str,
        system_message: str = "",
        user_message: str = "",
        temperature: float = 0.8,
        api_key: str = "",
        timeout: int = 600,
        n_predict: int = -1,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        seed: int = -1,
        dynatemp_range: float = 0.0,
        dynatemp_exponent: float = 1.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.1,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = -1,
        dry_sequence_breakers: str = '["\\n", ":", "\\"", "*"]',
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        typical_p: float = 1.0,
        n_keep: int = 0,
        stop_sequences: str = "[]",
        ignore_eos: bool = False,
        stream: bool = False,
        n_probs: int = 0,
        min_keep: int = 0,
        post_sampling_probs: bool = False,
        return_tokens: bool = False,
        timings_per_token: bool = False,
        grammar: str = "",
        json_schema: str = "",
        logit_bias: str = "[]",
        cache_prompt: bool = True,
        id_slot: int = -1,
        samplers: str = '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]',
        t_max_predict_ms: int = 0,
        messages: str = "[]",
        assistant_message: str = "",
        max_tokens: int = -1,
        model: str = "",
        tools: str = "[]",
        tool_choice: str = "auto",
        response_format: str = "",
        input_text: str = "",
        encoding_format: str = "float",
        embd_normalize: int = 2,
        content: str = "",
        tokens: str = "[]",
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = False,
        input_prefix: str = "",
        input_suffix: str = "",
        input_extra: str = "[]",
        query: str = "",
        documents: str = "[]",
        top_n: int = 10,
        lora: str = "[]",
        response_fields: str = "[]",
        image_data: str = "[]",
        images: Optional[torch.Tensor] = None,
        moe_mode: bool = False,
        extract_metadata: bool = True,
        debug_mode: bool = True,
    ) -> Tuple[str, str, str, int, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}

        if debug_mode:
            set_debug_mode(True)

        client = LlamaCppAPIClient(server_url, api_key=api_key, timeout=timeout)

        try:
            response_text = ""
            raw_response = ""
            error = ""
            status_code = 200
            metadata_list = []

            sampling_params: SamplingParams = {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "seed": seed,
                "repeat_penalty": repeat_penalty,
                "repeat_last_n": repeat_last_n,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "dry_multiplier": dry_multiplier,
                "dry_base": dry_base,
                "dry_allowed_length": dry_allowed_length,
                "dry_penalty_last_n": dry_penalty_last_n,
                "dry_sequence_breakers": dry_sequence_breakers,
                "mirostat": mirostat,
                "mirostat_tau": mirostat_tau,
                "mirostat_eta": mirostat_eta,
                "typical_p": typical_p,
                "n_keep": n_keep,
                "stop_sequences": stop_sequences,
                "ignore_eos": ignore_eos,
                "stream": stream,
                "n_probs": n_probs,
                "min_keep": min_keep,
                "post_sampling_probs": post_sampling_probs,
                "return_tokens": return_tokens,
                "timings_per_token": timings_per_token,
                "grammar": grammar,
                "logit_bias": logit_bias,
                "cache_prompt": cache_prompt,
                "id_slot": id_slot,
                "samplers": samplers,
                "t_max_predict_ms": t_max_predict_ms,
                "lora": lora,
                "dynatemp_range": dynatemp_range,
                "dynatemp_exponent": dynatemp_exponent,
                "xtc_probability": xtc_probability,
                "xtc_threshold": xtc_threshold,
            }

            # Apply MoE Presets
            if moe_mode:
                log_info("MoE optimization mode enabled")
                sampling_params["timings_per_token"] = False
                sampling_params["n_probs"] = 0
                sampling_params["post_sampling_probs"] = False
                # Simplify samplers for MoE if default
                if samplers == '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]':
                    sampling_params["samplers"] = '["top_k", "top_p", "temperature"]'

            api_response: ApiResponse

            if endpoint == "completion":
                c_params: CompletionParams = {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "sampling": sampling_params,
                }
                kwargs = self._build_completion_kwargs(c_params)
                api_response = client.handle_completion(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            elif endpoint == "chat_completions":
                chat_params: ChatParams = {
                    "messages": messages,
                    "system_message": system_message,
                    "user_message": user_message,
                    "prompt": prompt,
                    "assistant_message": assistant_message,
                    "max_tokens": max_tokens,
                    "model": model,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "response_format": response_format,
                    "image_data": image_data,
                    "images": images,
                    "extract_metadata": extract_metadata,
                    "sampling": sampling_params,
                }
                kwargs = self._build_chat_kwargs(chat_params)
                api_response = client.handle_chat_completions(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code
                metadata_list = api_response.metadata

            elif endpoint == "embeddings":
                kwargs = {
                    "input_text": input_text,
                    "model": model,
                    "encoding_format": encoding_format,
                }
                api_response = client.handle_embeddings(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            elif endpoint == "tokenize":
                kwargs = {
                    "content": content,
                    "add_special": add_special,
                    "parse_special": parse_special,
                    "with_pieces": with_pieces,
                }
                api_response = client.handle_tokenize(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            elif endpoint == "detokenize":
                kwargs = {
                    "tokens": tokens,
                }
                api_response = client.handle_detokenize(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            elif endpoint == "apply_template":
                kwargs = {
                    "messages": messages,
                }
                api_response = client.handle_apply_template(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            elif endpoint == "infill":
                i_params: InfillParams = {
                    "input_prefix": input_prefix,
                    "input_suffix": input_suffix,
                    "input_extra": input_extra,
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "sampling": sampling_params,
                }
                kwargs = self._build_infill_kwargs(i_params)
                api_response = client.handle_infill(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            elif endpoint == "reranking":
                r_params: RerankingParams = {
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                }
                kwargs = {
                    "model": r_params["model"],
                    "query": r_params["query"],
                    "documents": parse_json_param(r_params["documents"], []),
                    "top_n": r_params["top_n"],
                }
                api_response = client.handle_reranking(**kwargs)
                response = api_response.data
                raw_response = api_response.raw
                error = api_response.error
                status_code = api_response.status_code

            response_text = self._extract_response_text(response, endpoint)

            if metadata_list:
                for i, meta in enumerate(metadata_list):
                    metadata[f"image_{i}"] = meta

            # Extract timings
            if isinstance(response, dict) and "timings" in response:
                metadata["timings"] = response["timings"]

            if error:
                log_error(f"API Error: {error}")

        except Exception as e:
            log_error(f"Exception during request: {e}")
            response_text = ""
            raw_response = ""
            error = str(e)
            status_code = 500
        finally:
            client.close()

        node_response = self._build_node_response(
            response_text=response_text,
            raw_response=raw_response,
            error=error,
            status_code=status_code,
            metadata=metadata,
        )
        return (
            node_response["response"],
            node_response["raw_response"],
            node_response["error"],
            node_response["status_code"],
            node_response["metadata"],
        )


NODE_CLASS_MAPPINGS = {
    "LlamaCppClientNode": LlamaCppClientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {"LlamaCppClientNode": "LlamaCpp Client (Multimodal)"}
