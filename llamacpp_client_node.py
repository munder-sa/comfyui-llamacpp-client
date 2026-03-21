import json
from typing import Any, Dict, Tuple

try:
    from utils.llama_client import LlamaCppAPIClient
    from utils.logger import set_debug_mode, log_debug, log_error
    from utils.image_utils import (
        build_vision_content,
        DEFAULT_JPEG_QUALITY,
        extract_image_metadata,
        detect_image_format,
        extract_tensor_metadata,
    )
except ImportError:
    from .utils.llama_client import LlamaCppAPIClient
    from .utils.logger import set_debug_mode, log_debug, log_error
    from .utils.image_utils import (
        build_vision_content,
        DEFAULT_JPEG_QUALITY,
        extract_image_metadata,
        detect_image_format,
        extract_tensor_metadata,
    )


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
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The prompt text for completion/chat",
                    },
                ),
            },
            "optional": {
                "system_message": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "System message for chat"},
                ),
                "user_message": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "User message for chat"},
                ),
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
                # Connection & Auth
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
                # Core Generation Parameters
                "n_predict": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 1000000,
                        "tooltip": "Number of tokens to predict (-1 = infinity)",
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
                        "tooltip": "Top-p (nucleus) sampling",
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
                # Dynamic Temperature
                "dynatemp_range": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Dynamic temperature range",
                    },
                ),
                "dynatemp_exponent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Dynamic temperature exponent",
                    },
                ),
                # XTC Sampling
                "xtc_probability": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "XTC probability",
                    },
                ),
                "xtc_threshold": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "XTC threshold",
                    },
                ),
                # Repetition Control
                "repeat_penalty": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Repetition penalty",
                    },
                ),
                "repeat_last_n": (
                    "INT",
                    {
                        "default": 64,
                        "min": -1,
                        "max": 2048,
                        "tooltip": "Last n tokens for repetition penalty",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Presence penalty",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Frequency penalty",
                    },
                ),
                # DRY Sampling
                "dry_multiplier": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "DRY sampling multiplier",
                    },
                ),
                "dry_base": (
                    "FLOAT",
                    {
                        "default": 1.75,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "DRY sampling base value",
                    },
                ),
                "dry_allowed_length": (
                    "INT",
                    {"default": 2, "min": 1, "max": 100, "tooltip": "DRY allowed length"},
                ),
                "dry_penalty_last_n": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2048, "tooltip": "DRY penalty last n tokens"},
                ),
                "dry_sequence_breakers": (
                    "STRING",
                    {
                        "default": '["\\n", ":", "\\"", "*"]',
                        "multiline": False,
                        "tooltip": "JSON array of DRY sequence breakers",
                    },
                ),
                # Mirostat
                "mirostat": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2,
                        "tooltip": "Mirostat sampling (0=disabled, 1=v1, 2=v2)",
                    },
                ),
                "mirostat_tau": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.1,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Mirostat target entropy",
                    },
                ),
                "mirostat_eta": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Mirostat learning rate",
                    },
                ),
                # Other Sampling
                "typical_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Locally typical sampling",
                    },
                ),
                # Control Parameters
                "n_keep": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 2048,
                        "tooltip": "Number of tokens to keep from prompt",
                    },
                ),
                "stop_sequences": (
                    "STRING",
                    {"default": "[]", "multiline": True, "tooltip": "JSON array of stop sequences"},
                ),
                "ignore_eos": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Ignore end-of-stream token"},
                ),
                # Streaming and Output
                "stream": ("BOOLEAN", {"default": False, "tooltip": "Enable streaming mode"}),
                "n_probs": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Return top N token probabilities",
                    },
                ),
                "min_keep": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Minimum tokens to keep in sampler",
                    },
                ),
                "post_sampling_probs": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Return post-sampling probabilities"},
                ),
                "return_tokens": ("BOOLEAN", {"default": False, "tooltip": "Return raw token IDs"}),
                "timings_per_token": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Include timing information"},
                ),
                # Grammar and JSON
                "grammar": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "BNF grammar for constrained generation",
                    },
                ),
                "json_schema": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "JSON schema for constrained generation",
                    },
                ),
                # Logit Bias
                "logit_bias": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of logit bias modifications",
                    },
                ),
                # Cache and Slot Management
                "cache_prompt": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Re-use KV cache from previous requests"},
                ),
                "id_slot": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Assign to specific slot (-1 = auto)",
                    },
                ),
                # Sampler Order
                "samplers": (
                    "STRING",
                    {
                        "default": '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]',
                        "multiline": False,
                        "tooltip": "JSON array defining sampler order",
                    },
                ),
                # Timing Constraints
                "t_max_predict_ms": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60000,
                        "tooltip": "Maximum prediction time in milliseconds",
                    },
                ),
                # Chat-specific parameters
                "messages": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of chat messages (for chat_completions endpoint)",
                    },
                ),
                "assistant_message": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Assistant message for chat (for prefilling)",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 1000000,
                        "tooltip": "Maximum tokens in response (OpenAI style)",
                    },
                ),
                "model": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Model name/alias"},
                ),
                # Function calling
                "tools": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of available tools/functions",
                    },
                ),
                "tool_choice": (
                    "STRING",
                    {"default": "auto", "multiline": False, "tooltip": "Tool choice strategy"},
                ),
                # Response format
                "response_format": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "JSON object defining response format",
                    },
                ),
                # Embeddings-specific
                "input_text": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Input text for embeddings"},
                ),
                "encoding_format": (
                    ["float", "base64"],
                    {"default": "float", "tooltip": "Encoding format for embeddings"},
                ),
                "embd_normalize": (
                    "INT",
                    {"default": 2, "min": -1, "max": 10, "tooltip": "Embedding normalization type"},
                ),
                # Tokenization
                "content": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Content to tokenize/detokenize"},
                ),
                "tokens": (
                    "STRING",
                    {"default": "[]", "multiline": False, "tooltip": "JSON array of token IDs"},
                ),
                "add_special": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Add special tokens during tokenization"},
                ),
                "parse_special": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Parse special tokens during tokenization"},
                ),
                "with_pieces": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Return token pieces with IDs"},
                ),
                # Infill-specific
                "input_prefix": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Code prefix for infill"},
                ),
                "input_suffix": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Code suffix for infill"},
                ),
                "input_extra": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of additional context files",
                    },
                ),
                # Reranking-specific
                "query": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Query for reranking"},
                ),
                "documents": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of documents to rank",
                    },
                ),
                "top_n": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of top results to return",
                    },
                ),
                # LoRA adapters
                "lora": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of LoRA adapter configurations",
                    },
                ),
                # Response fields selection
                "response_fields": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": False,
                        "tooltip": "JSON array of specific response fields to return",
                    },
                ),
                # Multimodal support
                "image_data": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON array of image data objects",
                    },
                ),
                "images": (
                    "IMAGE",
                    {"tooltip": "ComfyUI image tensor to send to the multimodal model"},
                ),
                # Metadata extraction
                "extract_metadata": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Extract and return image metadata"},
                ),
                # Debugging
                "debug_mode": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable detailed debug logging to the console"},
                ),
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
        # Optional parameters - must match INPUT_TYPES order
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
        """
        Process a request to the llama-server and return the response.
        
        Returns:
            Tuple of (response, raw_response, error, status_code, metadata)
        """
        # Initialize metadata dictionary
        metadata: Dict[str, Any] = {}
        
        # Set debug mode
        if debug_mode:
            set_debug_mode(True)
        
        # Initialize client
        client = LlamaCppAPIClient(server_url, api_key=api_key, timeout=timeout)
        
        # Build request payload based on endpoint
        payload: Dict[str, Any] = {}
        
        # Handle multimodal content if images are provided
        if images is not None:
            # Process images and build vision content
            try:
                # Build vision content for multimodal models
                user_text = prompt or ""
                vision_content, img_metadata_list = build_vision_content(
                    user_text=user_text,
                    image_data=[],  # No JSON image data
                    tensor_images=images,
                    jpeg_quality=DEFAULT_JPEG_QUALITY,
                    extract_metadata=extract_metadata,
                )
                
                # Collect metadata from build_vision_content
                if img_metadata_list:
                    for i, meta in enumerate(img_metadata_list):
                        metadata[f"image_{i}"] = meta
                
                # Add vision content to payload for chat_completions endpoint
                if endpoint == "chat_completions":
                    payload["messages"] = [
                        {
                            "role": "user",
                            "content": vision_content
                        }
                    ]
                else:
                    # For other endpoints, use prompt as text
                    payload["prompt"] = prompt
            except Exception as e:
                log_error(f"Error processing images: {e}", e)
                return "", "", str(e), 500, metadata
        
        # Build payload based on endpoint
        if endpoint == "completion":
            payload["prompt"] = prompt
            payload["n_predict"] = n_predict
            payload["temperature"] = temperature
            payload["top_k"] = top_k
            payload["top_p"] = top_p
            payload["min_p"] = min_p
            payload["seed"] = seed
            payload["repeat_penalty"] = repeat_penalty
            payload["repeat_last_n"] = repeat_last_n
            payload["presence_penalty"] = presence_penalty
            payload["frequency_penalty"] = frequency_penalty
            payload["stop"] = json.loads(stop_sequences) if stop_sequences else []
            payload["stream"] = stream
            payload["cache_prompt"] = cache_prompt
            payload["id_slot"] = id_slot
            payload["samplers"] = json.loads(samplers) if samplers else []
            payload["t_max_predict_ms"] = t_max_predict_ms
            payload["grammar"] = grammar
            payload["logit_bias"] = json.loads(logit_bias) if logit_bias else []
            payload["n_probs"] = n_probs
            payload["min_keep"] = min_keep
            payload["post_sampling_probs"] = post_sampling_probs
            payload["return_tokens"] = return_tokens
            payload["timings_per_token"] = timings_per_token
            payload["ignore_eos"] = ignore_eos
            payload["n_keep"] = n_keep
            payload["dynatemp_range"] = dynatemp_range
            payload["dynatemp_exponent"] = dynatemp_exponent
            payload["xtc_probability"] = xtc_probability
            payload["xtc_threshold"] = xtc_threshold
            payload["mirostat"] = mirostat
            payload["mirostat_tau"] = mirostat_tau
            payload["mirostat_eta"] = mirostat_eta
            payload["typical_p"] = typical_p
            payload["dry_multiplier"] = dry_multiplier
            payload["dry_base"] = dry_base
            payload["dry_allowed_length"] = dry_allowed_length
            payload["dry_penalty_last_n"] = dry_penalty_last_n
            payload["dry_sequence_breakers"] = json.loads(dry_sequence_breakers) if dry_sequence_breakers else []
            payload["lora"] = json.loads(lora) if lora else []
            
        elif endpoint == "chat_completions":
            # Use vision content if images were provided
            if images is not None and "messages" in payload:
                pass  # Messages already set from vision content
            else:
                payload["messages"] = json.loads(messages) if messages else []
                if assistant_message:
                    payload["messages"].append({"role": "assistant", "content": assistant_message})
            payload["max_tokens"] = max_tokens
            payload["model"] = model
            payload["temperature"] = temperature
            payload["top_k"] = top_k
            payload["top_p"] = top_p
            payload["min_p"] = min_p
            payload["seed"] = seed
            payload["stop"] = json.loads(stop_sequences) if stop_sequences else []
            payload["stream"] = stream
            payload["tools"] = json.loads(tools) if tools else []
            payload["tool_choice"] = tool_choice
            payload["response_format"] = json.loads(response_format) if response_format else None
            payload["grammar"] = grammar
            payload["logit_bias"] = json.loads(logit_bias) if logit_bias else []
            payload["n_probs"] = n_probs
            payload["min_keep"] = min_keep
            payload["post_sampling_probs"] = post_sampling_probs
            payload["return_tokens"] = return_tokens
            payload["timings_per_token"] = timings_per_token
            payload["ignore_eos"] = ignore_eos
            payload["dynatemp_range"] = dynatemp_range
            payload["dynatemp_exponent"] = dynatemp_exponent
            payload["xtc_probability"] = xtc_probability
            payload["xtc_threshold"] = xtc_threshold
            payload["repeat_penalty"] = repeat_penalty
            payload["repeat_last_n"] = repeat_last_n
            payload["presence_penalty"] = presence_penalty
            payload["frequency_penalty"] = frequency_penalty
            payload["mirostat"] = mirostat
            payload["mirostat_tau"] = mirostat_tau
            payload["mirostat_eta"] = mirostat_eta
            payload["typical_p"] = typical_p
            payload["lora"] = json.loads(lora) if lora else []
            
        elif endpoint == "embeddings":
            payload["input_text"] = input_text
            payload["encoding_format"] = encoding_format
            payload["embd_normalize"] = embd_normalize
            
        elif endpoint == "tokenize":
            payload["content"] = content
            payload["add_special"] = add_special
            payload["parse_special"] = parse_special
            payload["with_pieces"] = with_pieces
            
        elif endpoint == "detokenize":
            payload["tokens"] = json.loads(tokens) if tokens else []
            
        elif endpoint == "apply_template":
            payload["content"] = content
            payload["add_special"] = add_special
            
        elif endpoint == "infill":
            payload["input_prefix"] = input_prefix
            payload["input_suffix"] = input_suffix
            payload["input_extra"] = json.loads(input_extra) if input_extra else []
            payload["n_predict"] = n_predict
            payload["temperature"] = temperature
            payload["top_k"] = top_k
            payload["top_p"] = top_p
            payload["min_p"] = min_p
            payload["seed"] = seed
            payload["repeat_penalty"] = repeat_penalty
            payload["repeat_last_n"] = repeat_last_n
            payload["presence_penalty"] = presence_penalty
            payload["frequency_penalty"] = frequency_penalty
            payload["stop"] = json.loads(stop_sequences) if stop_sequences else []
            payload["stream"] = stream
            payload["cache_prompt"] = cache_prompt
            payload["id_slot"] = id_slot
            payload["samplers"] = json.loads(samplers) if samplers else []
            payload["t_max_predict_ms"] = t_max_predict_ms
            payload["grammar"] = grammar
            payload["logit_bias"] = json.loads(logit_bias) if logit_bias else []
            payload["n_probs"] = n_probs
            payload["min_keep"] = min_keep
            payload["post_sampling_probs"] = post_sampling_probs
            payload["return_tokens"] = return_tokens
            payload["timings_per_token"] = timings_per_token
            payload["ignore_eos"] = ignore_eos
            payload["n_keep"] = n_keep
            payload["dynatemp_range"] = dynatemp_range
            payload["dynatemp_exponent"] = dynatemp_exponent
            payload["xtc_probability"] = xtc_probability
            payload["xtc_threshold"] = xtc_threshold
            payload["mirostat"] = mirostat
            payload["mirostat_tau"] = mirostat_tau
            payload["mirostat_eta"] = mirostat_eta
            payload["typical_p"] = typical_p
            payload["lora"] = json.loads(lora) if lora else []
            
        elif endpoint == "reranking":
            payload["query"] = query
            payload["documents"] = json.loads(documents) if documents else []
            payload["top_n"] = top_n
            
        # Execute the request
        try:
            response = client.execute(endpoint, payload)
            status_code = 200
            raw_response = json.dumps(response) if isinstance(response, dict) else str(response)
            
            # Parse response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", ""))
            else:
                response_text = str(response)
                
        except Exception as e:
            response_text = ""
            raw_response = ""
            status_code = 500
            return "", "", str(e), status_code, metadata
        
        return response_text, raw_response, "", status_code, metadata


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "LlamaCppClientNode": LlamaCppClientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppClientNode": "LlamaCpp Client (Multimodal)"
}
