import json

# Set to True to enable debug logging, False for release
DEBUG_MODE = True
PREFIX = "[LlamaCppClientNode]"


def set_debug_mode(mode: bool):
    """Dynamically update the debug mode flag."""
    global DEBUG_MODE
    DEBUG_MODE = mode


def log_debug(message: str, data=None):
    """Log a debug message if DEBUG_MODE is enabled.
    If data is provided, it will be safely formatted to avoid massive base64 prints.
    """
    if not DEBUG_MODE:
        return

    if data is not None:
        formatted_data = _safe_format_data(data)
        print(f"{PREFIX} {message} | Data: {formatted_data}")
    else:
        print(f"{PREFIX} {message}")


def log_error(message: str, exception: Exception = None):
    """Always log errors regardless of DEBUG_MODE."""
    if exception:
        print(f"{PREFIX} ERROR: {message} | Exception: {str(exception)}")
    else:
        print(f"{PREFIX} ERROR: {message}")


def log_info(message: str):
    """Log general info messages. Can be toggled on/off with DEBUG_MODE or kept permanent."""
    if DEBUG_MODE:
        print(f"{PREFIX} INFO: {message}")


def _safe_format_data(data, max_str_len=100) -> str:
    """Helper to safely format large dictionaries/lists so base64 strings don't flood the terminal."""
    if isinstance(data, dict):
        safe_dict = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > max_str_len:
                safe_dict[k] = f"{v[:max_str_len]}... [truncated {len(v)} chars]"
            elif isinstance(v, (dict, list)):
                # Simplified recursive check
                safe_dict[k] = _safe_format_data(v, max_str_len)
            else:
                safe_dict[k] = v
        try:
            return json.dumps(safe_dict, indent=2)
        except:
            return str(safe_dict)

    elif isinstance(data, list):
        safe_list = []
        for v in data:
            if isinstance(v, str) and len(v) > max_str_len:
                safe_list.append(f"{v[:max_str_len]}... [truncated]")
            elif isinstance(v, (dict, list)):
                safe_list.append(_safe_format_data(v, max_str_len))
            else:
                safe_list.append(v)
        try:
            return json.dumps(safe_list, indent=2)
        except:
            return str(safe_list)

    elif isinstance(data, str) and len(data) > max_str_len:
        return f"{data[:max_str_len]}... [truncated {len(data)} chars]"

    return str(data)
