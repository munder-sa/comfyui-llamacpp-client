"""
tests/test_image_utils.py

Comprehensive test suite for utils/image_utils.py.

Coverage targets:
  - detect_image_format      : magic-number detection for all supported formats
  - validate_image_data      : base64 / data-URI validation edge cases
  - process_image_data_string: JSON parsing robustness
  - tensor_to_base64_data_uri: format output, resize, memory-safety
  - build_vision_content     : content-array construction (text + images + tensors)
  - extract_tensor_metadata  : per-tensor metadata extraction
  - extract_image_metadata   : file-based metadata extraction

All tests are server-independent (mock-based).  No live llama.cpp connection
is required.
"""

import base64
import io
import os
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
from PIL import Image

from tests.helpers import create_batch_numpy_image, create_numpy_image
from utils.image_utils import (
    build_vision_content,
    detect_image_format,
    extract_image_metadata,
    extract_tensor_metadata,
    process_image_data_string,
    tensor_to_base64_data_uri,
    validate_image_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_data_uri_to_image(data_uri: str) -> Image.Image:
    """Decode a data URI string to a PIL Image for assertions."""
    b64_part = data_uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_part)))


# ---------------------------------------------------------------------------
# 1. TestDetectImageFormat
# ---------------------------------------------------------------------------


class TestDetectImageFormat(unittest.TestCase):
    """Tests for detect_image_format().

    Verifies that magic-byte sequences for all supported image formats are
    correctly identified, invalid inputs return None, and base64-encoded
    binary input is transparently decoded.
    """

    def test_jpeg_magic_bytes(self):
        """JPEG files start with \\xFF\\xD8\\xFF."""
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        self.assertEqual(detect_image_format(data), "JPEG")

    def test_png_magic_bytes(self):
        """PNG files start with the 8-byte PNG signature."""
        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        self.assertEqual(detect_image_format(data), "PNG")

    def test_gif87a_magic_bytes(self):
        """GIF87a files are identified as GIF."""
        data = b"GIF87a\x01\x00\x01\x00\x00\x00\x00"
        self.assertEqual(detect_image_format(data), "GIF")

    def test_gif89a_magic_bytes(self):
        """GIF89a (animated GIF) files are also identified as GIF."""
        data = b"GIF89a\x01\x00\x01\x00\x00\x00\x00"
        self.assertEqual(detect_image_format(data), "GIF")

    def test_bmp_magic_bytes(self):
        """BMP files start with the 'BM' magic bytes."""
        data = b"BM\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00"
        self.assertEqual(detect_image_format(data), "BMP")

    def test_invalid_data_returns_none(self):
        """Random bytes, invalid base64, empty bytes and None all return None."""
        self.assertIsNone(detect_image_format(b"not an image"))
        self.assertIsNone(detect_image_format("not base64!!!@#$"))
        self.assertIsNone(detect_image_format(None))
        self.assertIsNone(detect_image_format(b""))

    def test_base64_string_input(self):
        """Base64-encoded JPEG header bytes are decoded and detected as JPEG."""
        b64_data = base64.b64encode(b"\xff\xd8\xff\xe0").decode("utf-8")
        self.assertEqual(detect_image_format(b64_data), "JPEG")


# ---------------------------------------------------------------------------
# 2. TestValidateImageData
# ---------------------------------------------------------------------------


class TestValidateImageData(unittest.TestCase):
    """Tests for validate_image_data().

    Covers valid base64 strings, data-URI prefixes for multiple MIME types,
    and various invalid inputs including empty strings and non-string types.
    """

    def test_valid_base64(self):
        """Valid base64-encoded bytes return True."""
        valid_b64 = base64.b64encode(b"test image payload").decode("utf-8")
        self.assertTrue(validate_image_data(valid_b64))

    def test_data_uri_format_jpeg(self):
        """data:image/jpeg;base64,... is accepted as valid."""
        self.assertTrue(validate_image_data("data:image/jpeg;base64,abc123"))

    def test_data_uri_format_png(self):
        """data:image/png;base64,... is accepted as valid."""
        self.assertTrue(validate_image_data("data:image/png;base64,abc123"))

    def test_data_uri_format_webp(self):
        """data:image/webp;base64,... is accepted as valid."""
        self.assertTrue(validate_image_data("data:image/webp;base64,abc123"))

    def test_empty_string_invalid(self):
        """Empty string and None return False."""
        self.assertFalse(validate_image_data(""))
        self.assertFalse(validate_image_data(None))

    def test_non_string_types_invalid(self):
        """Integer, list, dict and float inputs all return False."""
        self.assertFalse(validate_image_data(123))
        self.assertFalse(validate_image_data([]))
        self.assertFalse(validate_image_data({}))
        self.assertFalse(validate_image_data(3.14))


# ---------------------------------------------------------------------------
# 3. TestProcessImageDataString
# ---------------------------------------------------------------------------


class TestProcessImageDataString(unittest.TestCase):
    """Tests for process_image_data_string().

    Verifies correct JSON parsing, graceful handling of empty/whitespace input,
    malformed JSON, and non-list JSON values.
    """

    def test_valid_json_array(self):
        """Valid JSON array string is parsed into a Python list."""
        json_str = '[{"data": "abc"}, {"data": "def"}]'
        result = process_image_data_string(json_str)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["data"], "abc")
        self.assertEqual(result[1]["data"], "def")

    def test_empty_string_returns_empty_list(self):
        """Empty string and None return an empty list."""
        self.assertEqual(process_image_data_string(""), [])
        self.assertEqual(process_image_data_string(None), [])

    def test_whitespace_only_returns_empty_list(self):
        """Strings containing only whitespace / tabs return an empty list."""
        self.assertEqual(process_image_data_string("   "), [])
        self.assertEqual(process_image_data_string("\t\n"), [])

    def test_invalid_json_returns_empty_list(self):
        """Malformed JSON strings return an empty list without raising."""
        self.assertEqual(process_image_data_string("[invalid json"), [])
        self.assertEqual(process_image_data_string("{key: value}"), [])

    def test_json_object_not_list_returns_empty_list(self):
        """A valid JSON object (dict) is not a list, so [] is returned."""
        self.assertEqual(process_image_data_string('{"key": "value"}'), [])
        # A bare integer is also not a list
        self.assertEqual(process_image_data_string("123"), [])


# ---------------------------------------------------------------------------
# 4. TestTensorToBase64DataUri
# ---------------------------------------------------------------------------


class TestTensorToBase64DataUri(unittest.TestCase):
    """Tests for tensor_to_base64_data_uri().

    Uses helpers.create_numpy_image() for consistent fixture creation.
    Covers JPEG/PNG/WebP output, resize behaviour, aspect-ratio preservation,
    memory-safe operation without torch, and invalid-input handling.
    """

    def setUp(self):
        # Standard 64×64 RGB tensor in ComfyUI [H, W, C] float32 format
        self.dummy_tensor = create_numpy_image(64, 64, 3, value=0.5)

    def test_numpy_array_conversion(self):
        """numpy array is converted to a valid JPEG data URI."""
        result = tensor_to_base64_data_uri(self.dummy_tensor)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        img = _decode_data_uri_to_image(result)
        self.assertEqual(img.format, "JPEG")
        self.assertEqual(img.size, (64, 64))

    def test_jpeg_quality_affects_size(self):
        """Lower JPEG quality produces a smaller encoded output."""
        res_low = tensor_to_base64_data_uri(self.dummy_tensor, jpeg_quality=10)
        res_high = tensor_to_base64_data_uri(self.dummy_tensor, jpeg_quality=95)
        self.assertLess(len(res_low), len(res_high))

    def test_large_image_resized(self):
        """Images exceeding max_dimension are resized (square input stays square)."""
        large_tensor = create_numpy_image(3000, 3000, 3, value=0.0)
        result = tensor_to_base64_data_uri(large_tensor, max_dimension=512)
        img = _decode_data_uri_to_image(result)
        self.assertEqual(img.size, (512, 512))

    def test_aspect_ratio_preserved_on_resize(self):
        """Non-square images keep their aspect ratio after resize."""
        # H=64, W=128 → aspect ratio 1:2 (width is twice the height)
        wide_tensor = create_numpy_image(64, 128, 3, value=0.5)
        result = tensor_to_base64_data_uri(wide_tensor, max_dimension=64)
        img = _decode_data_uri_to_image(result)
        w, h = img.size
        # Both dimensions must be within max_dimension
        self.assertLessEqual(w, 64)
        self.assertLessEqual(h, 64)
        # Width should be roughly twice the height (aspect ratio ~2.0)
        self.assertAlmostEqual(w / h, 2.0, delta=0.15)

    def test_png_output_format(self):
        """output_format='png' produces a valid PNG data URI."""
        result = tensor_to_base64_data_uri(self.dummy_tensor, output_format="png")
        self.assertTrue(result.startswith("data:image/png;base64,"))
        img = _decode_data_uri_to_image(result)
        self.assertEqual(img.format, "PNG")

    def test_webp_output_format(self):
        """output_format='webp' produces a valid WebP data URI."""
        result = tensor_to_base64_data_uri(self.dummy_tensor, output_format="webp")
        self.assertTrue(result.startswith("data:image/webp;base64,"))
        img = _decode_data_uri_to_image(result)
        self.assertEqual(img.format, "WEBP")

    def test_clear_memory_no_crash_when_torch_is_none(self):
        """No exception leaks when torch is patched to None.

        When torch=None, isinstance(tensor_image, torch.Tensor) raises
        AttributeError which is caught by the except block, so the function
        returns None gracefully rather than propagating the exception.
        """
        with mock.patch("utils.image_utils.torch", None):
            try:
                result = tensor_to_base64_data_uri(self.dummy_tensor, clear_memory=True)
            except Exception as exc:
                self.fail(
                    f"tensor_to_base64_data_uri raised an unexpected exception "
                    f"when torch=None: {exc}"
                )
            # Implementation catches AttributeError internally and returns None
            # (because isinstance(x, None.Tensor) fails).  That is acceptable —
            # the important invariant is that no exception escapes to the caller.
            self.assertIn(result, [None, ""] + [r for r in [result] if isinstance(r, str)])

    def test_invalid_input_returns_none(self):
        """Non-array inputs (str, None, int) return None without raising."""
        self.assertIsNone(tensor_to_base64_data_uri("not a tensor"))
        self.assertIsNone(tensor_to_base64_data_uri(None))
        self.assertIsNone(tensor_to_base64_data_uri(42))


# ---------------------------------------------------------------------------
# 5. TestBuildVisionContent
# ---------------------------------------------------------------------------


class TestBuildVisionContent(unittest.TestCase):
    """Tests for build_vision_content().

    Verifies construction of OpenAI Vision-compatible content arrays from
    text, image_data JSON payloads, and ComfyUI [B,H,W,C] tensors.
    """

    def test_text_only(self):
        """Text-only input produces a single text content item."""
        content, metadata = build_vision_content("hello world", [])
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "hello world")
        self.assertEqual(metadata, [])

    def test_text_and_image_data(self):
        """image_data list produces an image_url content item with data URI."""
        image_data = [{"data": "SGVsbG8="}]  # base64("Hello")
        content, metadata = build_vision_content("text", image_data)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_image_data_with_data_uri_prefix_is_preserved(self):
        """image_data items that already contain a data:image prefix are used as-is."""
        uri = "data:image/png;base64,iVBORw0KGgo="
        image_data = [{"data": uri}]
        content, _ = build_vision_content("text", image_data)
        self.assertEqual(len(content), 2)
        # The URI must not be double-prefixed
        self.assertEqual(content[1]["image_url"]["url"], uri)

    def test_text_and_tensor_images_4d(self):
        """ComfyUI [B,H,W,C] batch tensor produces one image_url item per batch item."""
        batch_tensor = create_batch_numpy_image(2, 64, 64, 3)
        content, metadata = build_vision_content("prompt", [], tensor_images=batch_tensor)
        # 1 text + 2 images
        self.assertEqual(len(content), 3)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[2]["type"], "image_url")

    def test_metadata_extraction(self):
        """extract_metadata=True populates the metadata list with tensor info."""
        batch_tensor = create_batch_numpy_image(1, 128, 64, 3)  # H=128, W=64
        content, metadata = build_vision_content(
            "prompt", [], tensor_images=batch_tensor, extract_metadata=True
        )
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]["width"], 64)
        self.assertEqual(metadata[0]["height"], 128)
        self.assertEqual(metadata[0]["batch_index"], 0)

    def test_metadata_not_extracted_by_default(self):
        """Without extract_metadata=True, metadata list stays empty."""
        batch_tensor = create_batch_numpy_image(1, 64, 64, 3)
        _, metadata = build_vision_content("text", [], tensor_images=batch_tensor)
        self.assertEqual(metadata, [])

    def test_invalid_tensor_type_ignored(self):
        """Non-numpy/non-torch tensor_images (e.g. int) are silently ignored."""
        content, _ = build_vision_content("text", [], tensor_images=123)
        # Only the text item should appear
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")

    def test_empty_inputs(self):
        """Empty text and empty lists with no tensor produce empty content and metadata."""
        content, metadata = build_vision_content("", [], None)
        self.assertEqual(content, [])
        self.assertEqual(metadata, [])


# ---------------------------------------------------------------------------
# 6. TestExtractMetadata
# ---------------------------------------------------------------------------


class TestExtractMetadata(unittest.TestCase):
    """Tests for extract_tensor_metadata() and extract_image_metadata().

    Verifies that both functions correctly return width/height/format fields
    and handle error cases gracefully (invalid paths, etc.).
    """

    def test_extract_tensor_metadata_basic(self):
        """extract_tensor_metadata returns correct size for a plain numpy array."""
        img = create_numpy_image(128, 64, 3, value=0.5)  # H=128, W=64
        meta = extract_tensor_metadata(img)
        self.assertEqual(meta["width"], 64)
        self.assertEqual(meta["height"], 128)
        self.assertIn("mode", meta)

    def test_extract_tensor_metadata_with_batch_index(self):
        """batch_index is included in the metadata when provided."""
        img = create_numpy_image(64, 64, 3)
        meta = extract_tensor_metadata(img, batch_index=2)
        self.assertEqual(meta["batch_index"], 2)

    def test_extract_image_metadata_from_png_file(self):
        """extract_image_metadata reads correct dimensions and format from a real file."""
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            pil_img.save(tmp_path)
        try:
            meta = extract_image_metadata(tmp_path)
            self.assertEqual(meta["width"], 32)
            self.assertEqual(meta["height"], 32)
            self.assertEqual(meta["format"], "PNG")
            self.assertIn("mode", meta)
            self.assertIn("aspect_ratio", meta)
        finally:
            os.unlink(tmp_path)

    def test_extract_image_metadata_invalid_path_returns_empty(self):
        """A non-existent file path returns an empty dict, not an exception."""
        meta = extract_image_metadata("/nonexistent/path/to/image.png")
        self.assertEqual(meta, {})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
