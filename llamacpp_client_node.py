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
        # Debug: Print all input parameters to verify correct mapping
        # This ensures that external inputs (especially system_message) are correctly mapped to variables
        print("=" * 60)
        print("[DEBUG] process_request called with:")
        print(f"  server_url: {server_url}")
        print(f"  endpoint: {endpoint}")
        print(f"  prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  prompt: {prompt}")
        print(f"  system_message: {system_message[:50]}..." if len(system_message) > 50 else f"  system_message: {system_message}")
        print(f"  user_message: {user_message[:50]}..." if len(user_message) > 50 else f"  user_message: {user_message}")
        print(f"  temperature: {temperature}")
        print(f"  api_key: {api_key}")
        print(f"  timeout: {timeout}")
        print(f"  n_predict: {n_predict}")
        print(f"  top_k: {top_k}")
        print(f"  top_p: {top_p}")
        print(f"  min_p: {min_p}")
        print(f"  seed: {seed}")
        print(f"  dynatemp_range: {dynatemp_range}")
        print(f"  dynatemp_exponent: {dynatemp_exponent}")
        print(f"  xtc_probability: {xtc_probability}")
        print(f"  xtc_threshold: {xtc_threshold}")
        print(f"  repeat_penalty: {repeat_penalty}")
        print(f"  repeat_last_n: {repeat_last_n}")
        print(f"  presence_penalty: {presence_penalty}")
        print(f"  frequency_penalty: {frequency_penalty}")
        print(f"  dry_multiplier: {dry_multiplier}")
        print(f"  dry_base: {dry_base}")
        print(f"  dry_allowed_length: {dry_allowed_length}")
        print(f"  dry_penalty_last_n: {dry_penalty_last_n}")
        print(f"  dry_sequence_breakers: {dry_sequence_breakers}")
        print(f"  mirostat: {mirostat}")
        print(f"  mirostat_tau: {mirostat_tau}")
        print(f"  mirostat_eta: {mirostat_eta}")
        print(f"  typical_p: {typical_p}")
        print(f"  n_keep: {n_keep}")
        print(f"  stop_sequences: {stop_sequences}")
        print(f"  ignore_eos: {ignore_eos}")
        print(f"  stream: {stream}")
        print(f"  n_probs: {n_probs}")
        print(f"  min_keep: {min_keep}")
        print(f"  post_sampling_probs: {post_sampling_probs}")
        print(f"  return_tokens: {return_tokens}")
        print(f"  timings_per_token: {timings_per_token}")
        print(f"  grammar: {grammar[:50]}..." if len(grammar) > 50 else f"  grammar: {grammar}")
        print(f"  json_schema: {json_schema[:50]}..." if len(json_schema) > 50 else f"  json_schema: {json_schema}")
        print(f"  logit_bias: {logit_bias}")
        print(f"  cache_prompt: {cache_prompt}")
        print(f"  id_slot: {id_slot}")
        print(f"  samplers: {samplers}")
        print(f"  t_max_predict_ms: {t_max_predict_ms}")
        print(f"  messages: {messages[:50]}..." if len(messages) > 50 else f"  messages: {messages}")
        print(f"  assistant_message: {assistant_message[:50]}..." if len(assistant_message) > 50 else f"  assistant_message: {assistant_message}")
        print(f"  max_tokens: {max_tokens}")
        print(f"  model: {model}")
        print(f"  tools: {tools[:50]}..." if len(tools) > 50 else f"  tools: {tools}")
        print(f"  tool_choice: {tool_choice}")
        print(f"  response_format: {response_format[:50]}..." if len(response_format) > 50 else f"  response_format: {response_format}")
        print(f"  input_text: {input_text[:50]}..." if len(input_text) > 50 else f"  input_text: {input_text}")
        print(f"  encoding_format: {encoding_format}")
        print(f"  embd_normalize: {embd_normalize}")
        print(f"  content: {content[:50]}..." if len(content) > 50 else f"  content: {content}")
        print(f"  tokens: {tokens}")
        print(f"  add_special: {add_special}")
        print(f"  parse_special: {parse_special}")
        print(f"  with_pieces: {with_pieces}")
        print(f"  input_prefix: {input_prefix[:50]}..." if len(input_prefix) > 50 else f"  input_prefix: {input_prefix}")
        print(f"  input_suffix: {input_suffix[:50]}..." if len(input_suffix) > 50 else f"  input_suffix: {input_suffix}")
        print(f"  input_extra: {input_extra}")
        print(f"  query: {query[:50]}..." if len(query) > 50 else f"  query: {query}")
        print(f"  documents: {documents[:50]}..." if len(documents) > 50 else f"  documents: {documents}")
        print(f"  top_n: {top_n}")
        print(f"  lora: {lora[:50]}..." if len(lora) > 50 else f"  lora: {lora}")
        print(f"  response_fields: {response_fields}")
        print(f"  image_data: {image_data[:50]}..." if len(image_data) > 50 else f"  image_data: {image_data}")
        print(f"  images: {images}")
        print(f"  extract_metadata: {extract_metadata}")
        print(f"  debug_mode: {debug_mode}")
        print("=" * 60)
        
        # Initialize metadata dictionary
        metadata: Dict[str, Any] = {}
        
        # Set debug mode
        if debug_mode:
            set_debug_mode(True)
        
        # Initialize client
        client = LlamaCppAPIClient(server_url, api_key=api_key, timeout=timeout)
        
        # Execute the request based on endpoint
        try:
            response_text = ""
            raw_response = ""
            error = ""
            status_code = 200
            metadata_list = []
            
            if endpoint == "completion":
                # Build kwargs for handle_completion
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
                # Build messages list with proper handling for multimodal content
                # Using stack-based approach to guarantee system message is always first
                final_messages = []
                
                # [REQUIRED] System prompt placement (highest priority)
                if system_message:
                    final_messages.append({"role": "system", "content": system_message})
                
                # [OPTIONAL] History messages (prevent duplicate system prompts)
                if messages:
                    try:
                        history_messages = json.loads(messages)
                        for msg in history_messages:
                            # Prevent duplicate system prompt: skip existing system role
                            if msg.get("role") == "system" and system_message:
                                continue  # Prioritize external system_message
                            final_messages.append(msg)
                    except json.JSONDecodeError:
                        log_error("Invalid messages JSON format")
                
                # [REQUIRED] User message and vision content integration
                # Combine user_message and vision_content into a single user message if both exist
                combined_user_content = []
                
                # Add text content
                if user_message:
                    combined_user_content.append({"type": "text", "text": user_message})
                elif prompt:
                    combined_user_content.append({"type": "text", "text": prompt})
                
                # Add vision content if images are provided
                if images is not None:
                    try:
                        user_text = prompt or user_message or ""
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
                        
                        # Add vision content to combined user message
                        if isinstance(vision_content, list):
                            combined_user_content.extend(vision_content)
                        else:
                            combined_user_content.append({"type": "image_url", "image_url": {"url": vision_content}})
                    except Exception as e:
                        log_error(f"Error processing images: {e}", e)
                        return "", "", str(e), 500, metadata
                
                # Add combined user message if there's content
                if combined_user_content:
                    final_messages.append({"role": "user", "content": combined_user_content})
                
                # [OPTIONAL] Assistant message
                if assistant_message:
                    final_messages.append({"role": "assistant", "content": assistant_message})
                
                # [REQUIRED] Safety net: Ensure system message exists at index 0
                if not final_messages or final_messages[0].get("role") != "system":
                    log_debug("Safety net: Inserting empty system prompt as fallback")
                    final_messages.insert(0, {"role": "system", "content": ""})
                
                # [REQUIRED] Final verification - prove system message is at index 0
                if final_messages and final_messages[0].get("role") == "system":
                    log_debug("✓ System message is correctly positioned at index 0")
                else:
                    log_error("✗ System message is NOT at index 0! This will cause API errors.")
                
                # Build kwargs for handle_chat_completions (returns 5 values)
                kwargs = {
                    "messages": final_messages,
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
                # Handle chat_completions returns 5 values
                response, raw_response, error, status_code, metadata_list = client.handle_chat_completions(**kwargs)
                
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
            
            # Parse response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", ""))
            else:
                response_text = str(response)
            
            # Collect metadata from images
            if metadata_list:
                for i, meta in enumerate(metadata_list):
                    metadata[f"image_{i}"] = meta
            
            # Log error if present
            if error:
                log_error(f"API Error: {error}")
                
        except Exception as e:
            log_error(f"Exception during request: {e}")
            response_text = ""
            raw_response = ""
            error = str(e)
            status_code = 500
        
        return response_text, raw_response, error, status_code, metadata


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "LlamaCppClientNode": LlamaCppClientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppClientNode": "LlamaCpp Client (Multimodal)"
}
