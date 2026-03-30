import base64
import io
import unittest

import numpy as np
from PIL import Image

from utils.image_utils import (
    build_vision_content,
    detect_image_format,
    process_image_data_string,
    tensor_to_base64_data_uri,
    validate_image_data,
)


class TestDetectImageFormat(unittest.TestCase):
    def test_jpeg_magic_bytes(self):
        # JPEG starts with \xFF\xD8\xFF
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        self.assertEqual(detect_image_format(data), "JPEG")

    def test_png_magic_bytes(self):
        # PNG starts with \x89PNG\r\n\x1a\n
        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        self.assertEqual(detect_image_format(data), "PNG")

    def test_invalid_data_returns_none(self):
        self.assertIsNone(detect_image_format(b"not an image"))
        self.assertIsNone(detect_image_format("not base64"))
        self.assertIsNone(detect_image_format(None))
        self.assertIsNone(detect_image_format(b""))

    def test_base64_string_input(self):
        # Base64 for a small JPEG header
        b64_data = base64.b64encode(b"\xff\xd8\xff\xe0").decode("utf-8")
        self.assertEqual(detect_image_format(b64_data), "JPEG")


class TestValidateImageData(unittest.TestCase):
    def test_valid_base64(self):
        # Valid base64 encoding of "test"
        valid_b64 = base64.b64encode(b"test").decode("utf-8")
        self.assertTrue(validate_image_data(valid_b64))

    def test_data_uri_format(self):
        self.assertTrue(validate_image_data("data:image/jpeg;base64,..."))
        self.assertTrue(validate_image_data("data:image/png;base64,..."))

    def test_empty_string_invalid(self):
        self.assertFalse(validate_image_data(""))
        self.assertFalse(validate_image_data(None))
        self.assertFalse(validate_image_data(123))


class TestProcessImageDataString(unittest.TestCase):
    def test_valid_json_array(self):
        json_str = '[{"data": "abc"}, {"data": "def"}]'
        result = process_image_data_string(json_str)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["data"], "abc")

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(process_image_data_string(""), [])
        self.assertEqual(process_image_data_string("   "), [])
        self.assertEqual(process_image_data_string(None), [])

    def test_invalid_json_returns_empty_list(self):
        self.assertEqual(process_image_data_string("[invalid json"), [])
        self.assertEqual(process_image_data_string("123"), [])


class TestTensorToBase64DataUri(unittest.TestCase):
    def setUp(self):
        # Create a dummy 64x64 RGB tensor (ComfyUI format: [H, W, C])
        # Note: image_utils process_request handles batch in build_vision_content,
        # but tensor_to_base64_data_uri expects a single image [H, W, C]
        self.dummy_tensor = np.ones((64, 64, 3), dtype=np.float32) * 0.5

    def test_numpy_array_conversion(self):
        result = tensor_to_base64_data_uri(self.dummy_tensor)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))

        # Verify it's valid base64
        b64_part = result.split(",")[1]
        decoded = base64.b64decode(b64_part)
        self.assertTrue(len(decoded) > 0)

        # Verify it's actually a JPEG
        img = Image.open(io.BytesIO(decoded))
        self.assertEqual(img.format, "JPEG")
        self.assertEqual(img.size, (64, 64))

    def test_jpeg_quality_affects_size(self):
        res_low = tensor_to_base64_data_uri(self.dummy_tensor, jpeg_quality=10)
        res_high = tensor_to_base64_data_uri(self.dummy_tensor, jpeg_quality=95)
        self.assertLess(len(res_low), len(res_high))

    def test_large_image_resized(self):
        # Create a large 3000x3000 image
        large_tensor = np.zeros((3000, 3000, 3), dtype=np.float32)
        # Should be resized to max 2048 by default (or specified)
        result = tensor_to_base64_data_uri(large_tensor, max_dimension=512)
        b64_part = result.split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_part)))
        self.assertEqual(img.size, (512, 512))

    def test_png_output_format(self):
        result = tensor_to_base64_data_uri(self.dummy_tensor, output_format="png")
        self.assertTrue(result.startswith("data:image/png;base64,"))
        b64_part = result.split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_part)))
        self.assertEqual(img.format, "PNG")

    def test_invalid_input_returns_none(self):
        # Non-array input should handle gracefully
        self.assertIsNone(tensor_to_base64_data_uri("not a tensor"))


class TestBuildVisionContent(unittest.TestCase):
    def test_text_only(self):
        content, metadata = build_vision_content("hello world", [])
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "hello world")
        self.assertEqual(metadata, [])

    def test_text_and_image_data(self):
        image_data = [{"data": "SGVsbG8="}]  # "Hello" in b64
        content, metadata = build_vision_content("text", image_data)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_text_and_tensor_images_4d(self):
        # ComfyUI batch format: [B, H, W, C]
        batch_tensor = np.zeros((2, 64, 64, 3), dtype=np.float32)
        content, metadata = build_vision_content("prompt", [], tensor_images=batch_tensor)
        # 1 text + 2 images
        self.assertEqual(len(content), 3)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[2]["type"], "image_url")

    def test_metadata_extraction(self):
        batch_tensor = np.zeros((1, 128, 64, 3), dtype=np.float32)
        content, metadata = build_vision_content(
            "prompt", [], tensor_images=batch_tensor, extract_metadata=True
        )
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]["width"], 64)
        self.assertEqual(metadata[0]["height"], 128)
        self.assertEqual(metadata[0]["batch_index"], 0)

    def test_empty_inputs(self):
        content, metadata = build_vision_content("", [], None)
        self.assertEqual(content, [])
        self.assertEqual(metadata, [])


if __name__ == "__main__":
    unittest.main()
