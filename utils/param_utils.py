import json
from typing import Dict, Any
from .logger import log_debug, log_error

# Parameters that need to be parsed as JSON
JSON_PARAMETERS = [
    'stop_sequences', 'logit_bias', 'samplers', 'messages', 'tools', 
    'response_format', 'input_extra', 'documents', 'lora', 'response_fields',
    'image_data', 'dry_sequence_breakers', 'tokens'
]

# Common completion and sampling parameters mapping
COMMON_COMPLETION_PARAMS = {
    "n_predict": "n_predict",
    "temperature": "temperature", 
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "seed": "seed",
    "dynatemp_range": "dynatemp_range",
    "dynatemp_exponent": "dynatemp_exponent",
    "xtc_probability": "xtc_probability",
    "xtc_threshold": "xtc_threshold",
    "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "dry_multiplier": "dry_multiplier",
    "dry_base": "dry_base",
    "dry_allowed_length": "dry_allowed_length",
    "dry_penalty_last_n": "dry_penalty_last_n",
    "dry_sequence_breakers": "dry_sequence_breakers",
    "mirostat": "mirostat",
    "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta",
    "typical_p": "typical_p",
    "n_keep": "n_keep",
    "stop_sequences": "stop",
    "ignore_eos": "ignore_eos",
    "stream": "stream",
    "n_probs": "n_probs",
    "min_keep": "min_keep",
    "post_sampling_probs": "post_sampling_probs",
    "return_tokens": "return_tokens",
    "timings_per_token": "timings_per_token",
    "grammar": "grammar",
    "json_schema": "json_schema",
    "logit_bias": "logit_bias",
    "cache_prompt": "cache_prompt",
    "id_slot": "id_slot",
    "samplers": "samplers",
    "t_max_predict_ms": "t_max_predict_ms",
    "lora": "lora",
    "response_fields": "response_fields",
    "image_data": "image_data"
}

# Chat completion specific mapping
CHAT_COMPLETION_PARAMS = {
    "max_tokens": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "seed": "seed",
    "stream": "stream",
    "stop_sequences": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "tools": "tools",
    "tool_choice": "tool_choice",
    "response_format": "response_format",
    "n_probs": "logprobs",
    "image_data": "image_data"
}

def clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values and convert string parameters to appropriate types."""
    cleaned = {}
    
    for key, value in params.items():
        if value is None:
            continue
            
        # Handle string parameters that should be parsed as JSON
        if key in JSON_PARAMETERS:
            if isinstance(value, str) and value.strip():
                # For image_data, strip out newlines to ensure proper JSON parsing
                if key == 'image_data':
                    log_debug(f"Processing parameter '{key}'. Original string length: {len(value)}")
                    value = value.replace('\n', '').replace('\r', '')
                try:
                    parsed_value = json.loads(value)
                    cleaned[key] = parsed_value
                    if key == 'image_data':
                        log_debug(f"Successfully parsed '{key}' as JSON. Result type: {type(parsed_value)}")
                except json.JSONDecodeError as e:
                    log_error(f"Error parsing JSON for parameter '{key}'", e)
                    continue
            elif isinstance(value, (list, dict)):
                if key == 'image_data':
                    log_debug(f"Received '{key}' already as type {type(value)}.")
                # Value is already a list or dict, use as is
                cleaned[key] = value
        else:
            cleaned[key] = value
            
    return cleaned

def map_parameters(kwargs: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Map UI kwargs to API parameters based on mapping."""
    params = {}
    for param_key, api_key in mapping.items():
        if param_key in kwargs and kwargs[param_key] is not None:
            params[api_key] = kwargs[param_key]
    return params
