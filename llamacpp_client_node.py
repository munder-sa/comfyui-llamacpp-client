from .utils.llama_client import LlamaCppAPIClient
from .utils.logger import set_debug_mode

class LlamaCppClientNode:
    """
    ComfyUI custom node that acts as a client for llama-server from llama.cpp.
    Supports ALL possible parameters that llama-server accepts through its various endpoints.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {
                    "default": "http://127.0.0.1:8080",
                    "multiline": False,
                    "tooltip": "Base URL of the llama-server instance"
                }),
                "endpoint": (["completion", "chat_completions", "embeddings", "tokenize", "detokenize", "apply_template", "infill", "reranking"], {
                    "default": "completion",
                    "tooltip": "API endpoint to use"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The prompt text for completion/chat"
                }),
            },
            "optional": {
                # Connection & Auth
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API key for authentication (if required)"
                }),
                "timeout": ("INT", {
                    "default": 600,
                    "min": 1,
                    "max": 3600,
                    "tooltip": "Request timeout in seconds"
                }),
                
                # Core Generation Parameters
                "n_predict": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1000000,
                    "tooltip": "Number of tokens to predict (-1 = infinity)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Sampling temperature"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Top-k sampling"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-p (nucleus) sampling"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Min-p sampling"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31-1,
                    "tooltip": "Random seed (-1 for random)"
                }),
                
                # Dynamic Temperature
                "dynatemp_range": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Dynamic temperature range"
                }),
                "dynatemp_exponent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Dynamic temperature exponent"
                }),
                
                # XTC Sampling
                "xtc_probability": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "XTC probability"
                }),
                "xtc_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "XTC threshold"
                }),
                
                # Repetition Control
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Repetition penalty"
                }),
                "repeat_last_n": ("INT", {
                    "default": 64,
                    "min": -1,
                    "max": 2048,
                    "tooltip": "Last n tokens for repetition penalty"
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Presence penalty"
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Frequency penalty"
                }),
                
                # DRY Sampling
                "dry_multiplier": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "DRY sampling multiplier"
                }),
                "dry_base": ("FLOAT", {
                    "default": 1.75,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "DRY sampling base value"
                }),
                "dry_allowed_length": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "tooltip": "DRY allowed length"
                }),
                "dry_penalty_last_n": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2048,
                    "tooltip": "DRY penalty last n tokens"
                }),
                "dry_sequence_breakers": ("STRING", {
                    "default": '["\\n", ":", "\\"", "*"]',
                    "multiline": False,
                    "tooltip": "JSON array of DRY sequence breakers"
                }),
                
                # Mirostat
                "mirostat": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Mirostat sampling (0=disabled, 1=v1, 2=v2)"
                }),
                "mirostat_tau": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Mirostat target entropy"
                }),
                "mirostat_eta": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Mirostat learning rate"
                }),
                
                # Other Sampling
                "typical_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Locally typical sampling"
                }),
                
                # Control Parameters
                "n_keep": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 2048,
                    "tooltip": "Number of tokens to keep from prompt"
                }),
                "stop_sequences": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of stop sequences"
                }),
                "ignore_eos": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Ignore end-of-stream token"
                }),
                
                # Streaming and Output
                "stream": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable streaming mode"
                }),
                "n_probs": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Return top N token probabilities"
                }),
                "min_keep": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Minimum tokens to keep in sampler"
                }),
                "post_sampling_probs": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return post-sampling probabilities"
                }),
                "return_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return raw token IDs"
                }),
                "timings_per_token": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include timing information"
                }),
                
                # Grammar and JSON
                "grammar": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "BNF grammar for constrained generation"
                }),
                "json_schema": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON schema for constrained generation"
                }),
                
                # Logit Bias
                "logit_bias": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of logit bias modifications"
                }),
                
                # Cache and Slot Management
                "cache_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Re-use KV cache from previous requests"
                }),
                "id_slot": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": "Assign to specific slot (-1 = auto)"
                }),
                
                # Sampler Order
                "samplers": ("STRING", {
                    "default": '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]',
                    "multiline": False,
                    "tooltip": "JSON array defining sampler order"
                }),
                
                # Timing Constraints
                "t_max_predict_ms": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 60000,
                    "tooltip": "Maximum prediction time in milliseconds"
                }),
                
                # Chat-specific parameters
                "messages": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of chat messages (for chat_completions endpoint)"
                }),
                "system_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System message for chat"
                }),
                "user_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User message for chat"
                }),
                "assistant_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Assistant message for chat (for prefilling)"
                }),
                "max_tokens": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1000000,
                    "tooltip": "Maximum tokens in response (OpenAI style)"
                }),
                "model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Model name/alias"
                }),
                
                # Function calling
                "tools": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of available tools/functions"
                }),
                "tool_choice": ("STRING", {
                    "default": "auto",
                    "multiline": False,
                    "tooltip": "Tool choice strategy"
                }),
                
                # Response format
                "response_format": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON object defining response format"
                }),
                
                # Embeddings-specific
                "input_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Input text for embeddings"
                }),
                "encoding_format": (["float", "base64"], {
                    "default": "float",
                    "tooltip": "Encoding format for embeddings"
                }),
                "embd_normalize": ("INT", {
                    "default": 2,
                    "min": -1,
                    "max": 10,
                    "tooltip": "Embedding normalization type"
                }),
                
                # Tokenization
                "content": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Content to tokenize/detokenize"
                }),
                "tokens": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": "JSON array of token IDs"
                }),
                "add_special": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add special tokens during tokenization"
                }),
                "parse_special": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Parse special tokens during tokenization"
                }),
                "with_pieces": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return token pieces with IDs"
                }),
                
                # Infill-specific
                "input_prefix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Code prefix for infill"
                }),
                "input_suffix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Code suffix for infill"
                }),
                "input_extra": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of additional context files"
                }),
                
                # Reranking-specific
                "query": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Query for reranking"
                }),
                "documents": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of documents to rank"
                }),
                "top_n": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of top results to return"
                }),
                
                # LoRA adapters
                "lora": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of LoRA adapter configurations"
                }),
                
                # Response fields selection
                "response_fields": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": "JSON array of specific response fields to return"
                }),
                
                # Multimodal support
                "image_data": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of image data objects"
                }),
                "images": ("IMAGE", {
                    "tooltip": "ComfyUI image tensor to send to the multimodal model"
                }),
                
                # Debugging
                "debug_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable detailed debug logging to the console"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("response", "raw_response", "error", "status_code")
    FUNCTION = "process_request"
    CATEGORY = "AI/LlamaCpp"
    
    def process_request(self, server_url: str, endpoint: str, prompt: str, **kwargs):
        """Process the request to llama-server via the API client."""
        try:
            # Set debug mode dynamically based on UI toggle
            is_debug = kwargs.get("debug_mode", True)
            set_debug_mode(is_debug)
            
            client = LlamaCppAPIClient(
                base_url=server_url,
                api_key=kwargs.get("api_key", ""),
                timeout=kwargs.get("timeout", 600)
            )
            
            if endpoint == "completion":
                response, raw_response, error, status_code = client.handle_completion(prompt, **kwargs)
            elif endpoint == "chat_completions":
                response, raw_response, error, status_code = client.handle_chat_completions(prompt=prompt, **kwargs)
            elif endpoint == "embeddings":
                response, raw_response, error, status_code = client.handle_embeddings(prompt=prompt, **kwargs)
            elif endpoint == "tokenize":
                response, raw_response, error, status_code = client.handle_tokenize(prompt=prompt, **kwargs)
            elif endpoint == "detokenize":
                response, raw_response, error, status_code = client.handle_detokenize(**kwargs)
            elif endpoint == "apply_template":
                response, raw_response, error, status_code = client.handle_apply_template(**kwargs)
            elif endpoint == "infill":
                response, raw_response, error, status_code = client.handle_infill(prompt=prompt, **kwargs)
            elif endpoint == "reranking":
                response, raw_response, error, status_code = client.handle_reranking(**kwargs)
            else:
                return "", "", f"Unsupported endpoint: {endpoint}", 400
                
            return response, raw_response, error, status_code
            
        except Exception as e:
            return "", "", f"Error processing request: {str(e)}", 500

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LlamaCppClient": LlamaCppClientNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppClient": "Llama.cpp Server Client"
}
