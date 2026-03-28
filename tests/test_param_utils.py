"""
Unit tests for utils/param_utils.py

Covers:
- clean_params(): JSON parsing, None/empty removal, list passthrough, image_data newline stripping
- map_parameters(): key mapping, missing keys, None value skipping
- safe_convert_to_int(): normal conversion, float string, invalid input, min/max clipping
- safe_convert_to_float(): normal conversion, "randomize" sentinel, min/max clipping
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.param_utils import (  # Fixed module-level import  # noqa: E402
    CHAT_COMPLETION_PARAMS,
    COMMON_COMPLETION_PARAMS,
    JSON_PARAMETERS,
    clean_params,
    map_parameters,
    safe_convert_to_float,
    safe_convert_to_int,
)

# Removed duplicate import block
# Removed unmatched parenthesis

# ---------------------------------------------------------------------------
# TestCleanParams
# ---------------------------------------------------------------------------


class TestCleanParams(unittest.TestCase):
    """Tests for clean_params()."""

    def test_none_values_removed(self):
        """None values must be stripped from the output dict."""
        params = {"temperature": 0.7, "grammar": None, "seed": 42}
        result = clean_params(params)
        self.assertNotIn("grammar", result)
        self.assertIn("temperature", result)
        self.assertIn("seed", result)

    def test_empty_string_removed(self):
        """Empty strings (and whitespace-only) should be stripped."""
        params = {"grammar": "", "model": "   ", "temperature": 0.5}
        result = clean_params(params)
        self.assertNotIn("grammar", result)
        self.assertNotIn("model", result)
        self.assertIn("temperature", result)

    def test_json_string_parsed_stop_sequences(self):
        """stop_sequences (a JSON_PARAMETER) given as a JSON string should be parsed to a list."""
        params = {"stop_sequences": '["\\n", "User:"]'}
        result = clean_params(params)
        self.assertIn("stop_sequences", result)
        self.assertIsInstance(result["stop_sequences"], list)
        self.assertEqual(result["stop_sequences"], ["\n", "User:"])

    def test_json_string_parsed_logit_bias(self):
        """logit_bias given as a JSON string should be parsed to a list/dict."""
        params = {"logit_bias": '[{"token_id": 123, "bias": 5.0}]'}
        result = clean_params(params)
        self.assertIn("logit_bias", result)
        self.assertIsInstance(result["logit_bias"], list)

    def test_json_string_parsed_messages(self):
        """messages given as a JSON string should be parsed to a list."""
        payload = '[{"role": "user", "content": "Hello"}]'
        params = {"messages": payload}
        result = clean_params(params)
        self.assertIn("messages", result)
        self.assertIsInstance(result["messages"], list)
        self.assertEqual(len(result["messages"]), 1)

    def test_list_passthrough(self):
        """A value that is already a list should pass through unchanged."""
        original = ["\n", "END"]
        params = {"stop_sequences": original}
        result = clean_params(params)
        self.assertEqual(result["stop_sequences"], original)

    def test_dict_passthrough(self):
        """A value that is already a dict should pass through unchanged."""
        original = {"key": "value"}
        params = {"response_format": original}
        result = clean_params(params)
        self.assertEqual(result["response_format"], original)

    def test_invalid_json_fallback_to_empty_list(self):
        """Unparseable JSON string for a JSON_PARAMETER should fall back to []."""
        params = {"stop_sequences": "{invalid json!!!"}
        result = clean_params(params)
        self.assertIn("stop_sequences", result)
        self.assertEqual(result["stop_sequences"], [])

    def test_image_data_newline_stripped(self):
        """image_data string should have newlines removed before JSON parsing."""
        # Build a JSON array with embedded newlines
        raw = '[\n{"data": "abc"}\n]'
        params = {"image_data": raw}
        result = clean_params(params)
        self.assertIn("image_data", result)
        self.assertIsInstance(result["image_data"], list)
        self.assertEqual(len(result["image_data"]), 1)

    def test_non_json_param_passthrough(self):
        """Non-JSON parameters (e.g. temperature) should be passed through as-is."""
        params = {"temperature": 0.9, "seed": 42, "n_predict": 100}
        result = clean_params(params)
        self.assertEqual(result["temperature"], 0.9)
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["n_predict"], 100)

    def test_boolean_passthrough(self):
        """Boolean values should be preserved."""
        params = {"stream": True, "cache_prompt": False}
        result = clean_params(params)
        self.assertIs(result["stream"], True)
        self.assertIs(result["cache_prompt"], False)

    def test_empty_list_kept(self):
        """An already-empty list should be kept (not dropped)."""
        params = {"stop_sequences": []}
        result = clean_params(params)
        self.assertIn("stop_sequences", result)
        self.assertEqual(result["stop_sequences"], [])

    def test_documents_list_passthrough(self):
        """documents given as a list passes through unchanged."""
        docs = ["Doc A", "Doc B"]
        params = {"documents": docs}
        result = clean_params(params)
        self.assertEqual(result["documents"], docs)

    def test_tokens_json_string_parsed(self):
        """tokens JSON string should be parsed to a list of ints."""
        params = {"tokens": "[1, 2, 3]"}
        result = clean_params(params)
        self.assertIsInstance(result["tokens"], list)
        self.assertEqual(result["tokens"], [1, 2, 3])


# ---------------------------------------------------------------------------
# TestMapParameters
# ---------------------------------------------------------------------------


class TestMapParameters(unittest.TestCase):
    """Tests for map_parameters()."""

    def test_basic_mapping(self):
        """Keys in kwargs that appear in mapping should be renamed."""
        kwargs = {"stop_sequences": ["END"], "temperature": 0.8}
        mapping = {"stop_sequences": "stop", "temperature": "temperature"}
        result = map_parameters(kwargs, mapping)
        self.assertIn("stop", result)
        self.assertIn("temperature", result)
        self.assertNotIn("stop_sequences", result)
        self.assertEqual(result["stop"], ["END"])

    def test_missing_keys_skipped(self):
        """kwargs keys not in mapping should produce no output."""
        kwargs = {"unknown_param": 42}
        mapping = {"temperature": "temperature"}
        result = map_parameters(kwargs, mapping)
        self.assertNotIn("unknown_param", result)
        self.assertNotIn("temperature", result)

    def test_none_values_skipped(self):
        """None values in kwargs should not appear in the output."""
        kwargs = {"temperature": None, "seed": 42}
        mapping = {"temperature": "temperature", "seed": "seed"}
        result = map_parameters(kwargs, mapping)
        self.assertNotIn("temperature", result)
        self.assertIn("seed", result)

    def test_common_completion_params_keys_present(self):
        """COMMON_COMPLETION_PARAMS should map all expected sampling fields."""
        expected_keys = [
            "n_predict",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "seed",
            "repeat_penalty",
            "mirostat",
            "grammar",
        ]
        for key in expected_keys:
            self.assertIn(
                key, COMMON_COMPLETION_PARAMS, f"{key} missing from COMMON_COMPLETION_PARAMS"
            )

    def test_chat_completion_params_keys_present(self):
        """CHAT_COMPLETION_PARAMS should contain max_tokens and tools."""
        self.assertIn("max_tokens", CHAT_COMPLETION_PARAMS)
        self.assertIn("tools", CHAT_COMPLETION_PARAMS)

    def test_full_mapping_round_trip(self):
        """Using COMMON_COMPLETION_PARAMS to map a full set of kwargs."""
        kwargs = {
            "n_predict": 200,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "seed": 99,
        }
        result = map_parameters(kwargs, COMMON_COMPLETION_PARAMS)
        # COMMON_COMPLETION_PARAMS maps these 1-to-1
        self.assertEqual(result["n_predict"], 200)
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["seed"], 99)


# ---------------------------------------------------------------------------
# TestSafeConvertToInt
# ---------------------------------------------------------------------------


class TestSafeConvertToInt(unittest.TestCase):
    """Tests for safe_convert_to_int()."""

    def test_normal_int_conversion(self):
        self.assertEqual(safe_convert_to_int(5), 5)

    def test_float_value_truncated(self):
        self.assertEqual(safe_convert_to_int(3.9), 3)

    def test_float_string_conversion(self):
        """'1.0' should convert to 1."""
        self.assertEqual(safe_convert_to_int("1.0"), 1)

    def test_int_string_conversion(self):
        self.assertEqual(safe_convert_to_int("42"), 42)

    def test_invalid_string_returns_default(self):
        self.assertEqual(safe_convert_to_int("abc", default=0), 0)

    def test_none_returns_default(self):
        self.assertEqual(safe_convert_to_int(None, default=-1), -1)

    def test_empty_string_returns_default(self):
        self.assertEqual(safe_convert_to_int("", default=0), 0)

    def test_json_array_string_returns_default(self):
        """A string that looks like JSON array should return default."""
        self.assertEqual(safe_convert_to_int("[1, 2, 3]", default=0), 0)

    def test_below_min_returns_default(self):
        """Value below min_val should return default."""
        self.assertEqual(safe_convert_to_int(-5, default=0, min_val=0), 0)

    def test_above_max_returns_default(self):
        """Value above max_val should return default."""
        self.assertEqual(safe_convert_to_int(200, default=100, max_val=100), 100)

    def test_within_range_passes(self):
        self.assertEqual(safe_convert_to_int(50, default=0, min_val=0, max_val=100), 50)


# ---------------------------------------------------------------------------
# TestSafeConvertToFloat
# ---------------------------------------------------------------------------


class TestSafeConvertToFloat(unittest.TestCase):
    """Tests for safe_convert_to_float()."""

    def test_normal_float_conversion(self):
        self.assertAlmostEqual(safe_convert_to_float(0.7), 0.7)

    def test_int_to_float(self):
        self.assertAlmostEqual(safe_convert_to_float(1), 1.0)

    def test_string_float_conversion(self):
        self.assertAlmostEqual(safe_convert_to_float("0.95"), 0.95)

    def test_randomize_sentinel_returns_default(self):
        """The special string 'randomize' should return the default value."""
        self.assertAlmostEqual(safe_convert_to_float("randomize", default=0.8), 0.8)

    def test_none_returns_default(self):
        self.assertAlmostEqual(safe_convert_to_float(None, default=1.0), 1.0)

    def test_empty_string_returns_default(self):
        self.assertAlmostEqual(safe_convert_to_float("", default=0.5), 0.5)

    def test_invalid_string_returns_default(self):
        self.assertAlmostEqual(safe_convert_to_float("abc", default=0.0), 0.0)

    def test_below_min_returns_default(self):
        self.assertAlmostEqual(safe_convert_to_float(-1.0, default=0.0, min_val=0.0), 0.0)

    def test_above_max_returns_default(self):
        self.assertAlmostEqual(safe_convert_to_float(2.0, default=1.0, max_val=1.0), 1.0)

    def test_within_range_passes(self):
        self.assertAlmostEqual(
            safe_convert_to_float(0.5, default=0.0, min_val=0.0, max_val=1.0), 0.5
        )


# ---------------------------------------------------------------------------
# TestJsonParametersList
# ---------------------------------------------------------------------------


class TestJsonParametersList(unittest.TestCase):
    """Sanity checks on the JSON_PARAMETERS constant."""

    def test_expected_keys_present(self):
        """All key JSON parameters must be listed."""
        required = [
            "stop_sequences",
            "logit_bias",
            "samplers",
            "messages",
            "tools",
            "response_format",
            "documents",
            "lora",
            "image_data",
            "tokens",
        ]
        for key in required:
            self.assertIn(key, JSON_PARAMETERS, f"'{key}' missing from JSON_PARAMETERS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
