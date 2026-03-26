import json
from typing import Any, Dict, Tuple

try:
    from utils.image_utils import (
        DEFAULT_JPEG_QUALITY,
        build_vision_content,
        detect_image_format,
        extract_image_metadata,
        extract_tensor_metadata,
    )
    from utils.llama_client import LlamaCppAPIClient
    from utils.logger import log_debug, log_error, set_debug_mode
except ImportError:
    from .utils.image_utils import (
        DEFAULT_JPEG_QUALITY,
        build_vision_content,
        detect_image_format,
        extract_image_metadata,
        extract_tensor_metadata,
    )
    from .utils.llama_client import LlamaCppAPIClient
    from .utils.logger import log_debug, log_error, set_debug_mode


class LlamaCppClientNode:
    """
    ComfyUI custom node that acts as a client for llama-server from llama.cpp.
    Supports ALL possible parameters that llama-server accepts through its various endpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
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
            "optional": {
                # ========== 共通パラメータ（先頭）==========
                # 温度制御
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
                # 繰り返し制御
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
                # DRY 制御
                "dry_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "dry_base": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 5.0, "step": 0.01}),
                "dry_allowed_length": ("INT", {"default": 2, "min": 1, "max": 100}),
                "dry_penalty_last_n": ("INT", {"default": -1, "min": -1, "max": 2048}),
                "dry_sequence_breakers": (
                    "STRING",
                    {"default": '["\\n", ":", "\\"", "*"]', "multiline": False},
                ),
                # ミロスタット制御
                "mirostat": ("INT", {"default": 0, "min": 0, "max": 2}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "mirostat_eta": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001},
                ),
                # その他サンプリング
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
                # 通常制御
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
                # エンドポイント固有パラメータ（共通パラメータの後）
                # completion/infill 固有
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The prompt text for completion/chat",
                    },
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
                # chat_completions 固有
                "system_message": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "System message for chat"},
                ),
                "user_message": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "User message for chat"},
                ),
                "assistant_message": ("STRING", {"default": "", "multiline": True}),
                "messages": ("STRING", {"default": "[]", "multiline": True}),
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 1000000}),
                "model": ("STRING", {"default": "", "multiline": False}),
                "tools": ("STRING", {"default": "[]", "multiline": True}),
                "tool_choice": ("STRING", {"default": "auto", "multiline": False}),
                "response_format": ("STRING", {"default": "", "multiline": True}),
                # embeddings 固有
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "encoding_format": (["float", "base64"], {"default": "float"}),
                "embd_normalize": ("INT", {"default": 2, "min": -1, "max": 10}),
                # tokenize 固有
                "content": ("STRING", {"default": "", "multiline": True}),
                "tokens": ("STRING", {"default": "[]", "multiline": False}),
                "add_special": ("BOOLEAN", {"default": False}),
                "parse_special": ("BOOLEAN", {"default": True}),
                "with_pieces": ("BOOLEAN", {"default": False}),
                # detokenize 固有
                # apply_template 固有
                # infill 固有（completion と重複あり）
                "input_prefix": ("STRING", {"default": "", "multiline": True}),
                "input_suffix": ("STRING", {"default": "", "multiline": True}),
                "input_extra": ("STRING", {"default": "[]", "multiline": True}),
                # reranking 固有
                "query": ("STRING", {"default": "", "multiline": True}),
                "documents": ("STRING", {"default": "[]", "multiline": True}),
                "top_n": ("INT", {"default": 10, "min": 1, "max": 1000}),
                # 特殊パラメータ（最後）
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
                "extract_metadata": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "JSON")
    RETURN_NAMES = ("response", "raw_response", "error", "status_code", "metadata")
    FUNCTION = "process_request"
    CATEGORY = "AI/LlamaCpp"

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
        images: Any = None,
        extract_metadata: bool = True,
        debug_mode: bool = True,
    ) -> Tuple[str, str, str, int, Dict[str, Any]]:
        # Initialize metadata dictionary
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

            if endpoint == "completion":
                kwargs = {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "seed": seed,
                    "repeat_penalty": repeat_penalty,
                    "repeat_last_n": repeat_last_n,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "stop": json.loads(stop_sequences) if stop_sequences else [],
                    "stream": stream,
                    "cache_prompt": cache_prompt,
                    "id_slot": id_slot,
                    "samplers": json.loads(samplers) if samplers else [],
                    "t_max_predict_ms": t_max_predict_ms,
                    "grammar": grammar,
                    "logit_bias": json.loads(logit_bias) if logit_bias else [],
                    "n_probs": n_probs,
                    "min_keep": min_keep,
                    "post_sampling_probs": post_sampling_probs,
                    "return_tokens": return_tokens,
                    "timings_per_token": timings_per_token,
                    "ignore_eos": ignore_eos,
                    "n_keep": n_keep,
                    "dynatemp_range": dynatemp_range,
                    "dynatemp_exponent": dynatemp_exponent,
                    "xtc_probability": xtc_probability,
                    "xtc_threshold": xtc_threshold,
                    "mirostat": mirostat,
                    "mirostat_tau": mirostat_tau,
                    "mirostat_eta": mirostat_eta,
                    "typical_p": typical_p,
                    "lora": json.loads(lora) if lora else [],
                }
                response, raw_response, error, status_code = client.handle_completion(**kwargs)

            elif endpoint == "chat_completions":
                # LlamaCppAPIClient（llama_client.py）にメッセージの組み立てを丸投げする
                kwargs = {
                    "messages": messages,
                    "system_message": system_message,
                    "user_message": user_message,
                    "prompt": prompt,
                    "assistant_message": assistant_message,
                    "max_tokens": max_tokens,
                    "model": model,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "seed": seed,
                    "stop": json.loads(stop_sequences) if stop_sequences else [],
                    "stream": stream,
                    "tools": json.loads(tools) if tools else [],
                    "tool_choice": tool_choice,
                    "response_format": json.loads(response_format) if response_format else None,
                    "grammar": grammar,
                    "logit_bias": json.loads(logit_bias) if logit_bias else [],
                    "n_probs": n_probs,
                    "min_keep": min_keep,
                    "post_sampling_probs": post_sampling_probs,
                    "return_tokens": return_tokens,
                    "timings_per_token": timings_per_token,
                    "ignore_eos": ignore_eos,
                    "dynatemp_range": dynatemp_range,
                    "dynatemp_exponent": dynatemp_exponent,
                    "xtc_probability": xtc_probability,
                    "xtc_threshold": xtc_threshold,
                    "repeat_penalty": repeat_penalty,
                    "repeat_last_n": repeat_last_n,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "mirostat": mirostat,
                    "mirostat_tau": mirostat_tau,
                    "mirostat_eta": mirostat_eta,
                    "typical_p": typical_p,
                    "lora": json.loads(lora) if lora else [],
                    "image_data": image_data,
                    "images": images,
                    "extract_metadata": extract_metadata,
                }
                (
                    response,
                    raw_response,
                    error,
                    status_code,
                    metadata_list,
                ) = client.handle_chat_completions(**kwargs)

            elif endpoint == "embeddings":
                kwargs = {
                    "input_text": input_text,
                    "model": model,
                    "encoding_format": encoding_format,
                }
                response, raw_response, error, status_code = client.handle_embeddings(**kwargs)

            elif endpoint == "tokenize":
                kwargs = {
                    "content": content,
                    "add_special": add_special,
                    "parse_special": parse_special,
                    "with_pieces": with_pieces,
                }
                response, raw_response, error, status_code = client.handle_tokenize(**kwargs)

            elif endpoint == "detokenize":
                kwargs = {
                    "tokens": tokens,
                }
                response, raw_response, error, status_code = client.handle_detokenize(**kwargs)

            elif endpoint == "apply_template":
                kwargs = {
                    "messages": messages,
                }
                response, raw_response, error, status_code = client.handle_apply_template(**kwargs)

            elif endpoint == "infill":
                kwargs = {
                    "input_prefix": input_prefix,
                    "input_suffix": input_suffix,
                    "input_extra": json.loads(input_extra) if input_extra else [],
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "seed": seed,
                    "repeat_penalty": repeat_penalty,
                    "repeat_last_n": repeat_last_n,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "stop": json.loads(stop_sequences) if stop_sequences else [],
                    "stream": stream,
                    "cache_prompt": cache_prompt,
                    "id_slot": id_slot,
                    "samplers": json.loads(samplers) if samplers else [],
                    "t_max_predict_ms": t_max_predict_ms,
                    "grammar": grammar,
                    "logit_bias": json.loads(logit_bias) if logit_bias else [],
                    "n_probs": n_probs,
                    "min_keep": min_keep,
                    "post_sampling_probs": post_sampling_probs,
                    "return_tokens": return_tokens,
                    "timings_per_token": timings_per_token,
                    "ignore_eos": ignore_eos,
                    "n_keep": n_keep,
                    "dynatemp_range": dynatemp_range,
                    "dynatemp_exponent": dynatemp_exponent,
                    "xtc_probability": xtc_probability,
                    "xtc_threshold": xtc_threshold,
                    "mirostat": mirostat,
                    "mirostat_tau": mirostat_tau,
                    "mirostat_eta": mirostat_eta,
                    "typical_p": typical_p,
                    "lora": json.loads(lora) if lora else [],
                }
                response, raw_response, error, status_code = client.handle_infill(**kwargs)

            elif endpoint == "reranking":
                kwargs = {
                    "model": model,
                    "query": query,
                    "documents": json.loads(documents) if documents else [],
                    "top_n": top_n,
                }
                response, raw_response, error, status_code = client.handle_reranking(**kwargs)

            # --- ここから修正：レスポンスからのテキスト抽出処理 ---
            if isinstance(response, dict):
                if endpoint == "chat_completions":
                    # チャットAPIの場合は ["choices"][0]["message"]["content"] を探す
                    choices = response.get("choices", [])
                    if choices and len(choices) > 0:
                        response_text = choices[0].get("message", {}).get("content", "")
                else:
                    # CompletionAPIなどの場合は直下の ["content"] か ["text"] を探す
                    response_text = response.get("content", response.get("text", ""))
            else:
                response_text = str(response) if response else ""

            # メタデータの回収
            if metadata_list:
                for i, meta in enumerate(metadata_list):
                    metadata[f"image_{i}"] = meta

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

        return response_text, raw_response, error, status_code, metadata


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "LlamaCppClientNode": LlamaCppClientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {"LlamaCppClientNode": "LlamaCpp Client (Multimodal)"}
