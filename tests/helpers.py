"""
Shared test helpers, mock factories, and mixins for the test suite.

All test modules should import common utilities from here to avoid duplication.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Mock HTTP Response Factory
# ---------------------------------------------------------------------------


def make_mock_response(
    status_code: int,
    json_data: Optional[Dict[str, Any]] = None,
    text: str = "",
    raise_json: bool = False,
) -> MagicMock:
    """Build a mock requests.Response object.

    Args:
        status_code: HTTP status code to simulate.
        json_data: Parsed JSON body (used when raise_json is False).
        text: Raw response text (used for error bodies).
        raise_json: If True, calling .json() raises JSONDecodeError.

    Returns:
        Configured MagicMock that behaves like requests.Response.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    if raise_json:
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    else:
        mock_resp.json.return_value = json_data if json_data is not None else {}
    return mock_resp


# ---------------------------------------------------------------------------
# Response Body Factories
# ---------------------------------------------------------------------------


def make_completion_response(content: str = "Hello from mock") -> Dict[str, Any]:
    """Return a minimal /completion endpoint response body.

    Args:
        content: The text content to include in the response.

    Returns:
        Dict matching llama-server /completion response format.
    """
    return {"content": content}


def make_chat_response(content: str = "Hello from mock chat") -> Dict[str, Any]:
    """Return a minimal /v1/chat/completions endpoint response body.

    Args:
        content: The assistant message content.

    Returns:
        Dict matching OpenAI-compatible chat completions response format.
    """
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "model": "mock-model",
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }


def make_tokenize_response(tokens: Optional[List[int]] = None) -> Dict[str, Any]:
    """Return a minimal /tokenize endpoint response body."""
    return {"tokens": tokens if tokens is not None else [1, 2, 3]}


def make_detokenize_response(text: str = "Hello world") -> Dict[str, Any]:
    """Return a minimal /detokenize endpoint response body."""
    return {"content": text}


def make_embeddings_response(dim: int = 4) -> Dict[str, Any]:
    """Return a minimal /v1/embeddings endpoint response body."""
    return {
        "data": [
            {
                "embedding": [0.1 * i for i in range(dim)],
                "index": 0,
                "object": "embedding",
            }
        ],
        "model": "mock-model",
    }


def make_apply_template_response(prompt: str = "formatted prompt") -> Dict[str, Any]:
    """Return a minimal /apply-template endpoint response body."""
    return {"prompt": prompt}


def make_reranking_response() -> Dict[str, Any]:
    """Return a minimal /reranking endpoint response body."""
    return {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]
    }


# ---------------------------------------------------------------------------
# Session Patch Mixin
# ---------------------------------------------------------------------------


class SessionPatchMixin:
    """Mixin that patches requests.Session so session.post() is controlled.

    Designed for use with unittest.TestCase subclasses.  Each test that needs
    HTTP control should call _patch_session_post() in the test body or setUp().
    """

    def _patch_session_post(
        self,
        client,
        side_effect=None,
        return_value=None,
    ) -> MagicMock:
        """Patch requests.Session and return the mock session instance.

        Args:
            client: The LlamaCppAPIClient instance whose session should be reset.
            side_effect: Optional side_effect for mock_session.post.
            return_value: Optional return_value for mock_session.post.

        Returns:
            The mock session instance (not the Session class mock).
        """
        patcher = patch("utils.llama_client.requests.Session")
        mock_session_cls = patcher.start()
        self.addCleanup(patcher.stop)  # type: ignore[attr-defined]

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        if side_effect is not None:
            mock_session.post.side_effect = side_effect
        if return_value is not None:
            mock_session.post.return_value = return_value

        # Force client to create a new Session on next request
        client._session = None
        return mock_session


# ---------------------------------------------------------------------------
# UI Widget / Node Mocks (shared by test_ui_logic.py tests)
# ---------------------------------------------------------------------------


class MockWidget:
    """Simulates a ComfyUI/LiteGraph node widget.

    Attributes:
        name: Widget parameter name.
        value: Current widget value.
        inputEl: Simulated DOM input element reference.
        element: Simulated DOM element reference.
        hidden: Current visibility state.
    """

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value
        self.inputEl: Optional["MockElement"] = None
        self.element: Optional["MockElement"] = None
        self.hidden: bool = False


class MockElement:
    """Simulates a minimal DOM element with style and hidden attributes."""

    def __init__(self) -> None:
        self.style: Dict[str, str] = {"display": "block"}
        self.hidden: bool = False


class MockNode:
    """Simulates a ComfyUI node with masterWidgets / widgets / inputs.

    Methods:
        add_widget: Append a MockWidget to both masterWidgets and widgets.
    """

    def __init__(self) -> None:
        self.masterWidgets: List[MockWidget] = []
        self.widgets: List[MockWidget] = []
        self.inputs: List[Optional[Dict[str, Any]]] = []

    def add_widget(self, name: str, value: Any) -> MockWidget:
        """Create and register a widget."""
        widget = MockWidget(name, value)
        self.masterWidgets.append(widget)
        self.widgets.append(widget)
        return widget


# ---------------------------------------------------------------------------
# Image / Tensor Helpers
# ---------------------------------------------------------------------------


def create_numpy_image(
    height: int = 256,
    width: int = 256,
    channels: int = 3,
    value: float = 0.5,
    dtype=np.float32,
) -> np.ndarray:
    """Create a solid-colour numpy image array in ComfyUI float32 (0-1) format.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of colour channels (3 for RGB).
        value: Pixel fill value (0.0–1.0).
        dtype: NumPy data type.

    Returns:
        NumPy array of shape (height, width, channels).
    """
    arr = np.full((height, width, channels), value, dtype=dtype)
    return arr


def create_batch_numpy_image(
    batch: int = 1,
    height: int = 256,
    width: int = 256,
    channels: int = 3,
    value: float = 0.5,
) -> np.ndarray:
    """Create a batch of solid-colour numpy images in ComfyUI [B,H,W,C] format.

    Args:
        batch: Batch size.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of colour channels.
        value: Pixel fill value (0.0–1.0).

    Returns:
        NumPy array of shape (batch, height, width, channels).
    """
    return np.full((batch, height, width, channels), value, dtype=np.float32)
