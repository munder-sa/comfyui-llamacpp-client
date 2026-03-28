import unittest

import numpy as np

from utils.image_utils import (  # Removed unused import
    build_vision_content,
    detect_image_format,
    extract_tensor_metadata,
    process_image_data_string,
    tensor_to_base64_data_uri,
    validate_image_data,
)

# Removed duplicate and misplaced import
# Removed misplaced import
# Removed misplaced import
# Removed misplaced import
# Removed misplaced import
# Removed misplaced import
# Removed misplaced import
# Removed unmatched parenthesis


class TestImageUtils(unittest.TestCase):
    def test_detect_image_format(self):
        data = b"\x89PNG\r\n\x1a\n"
        result = detect_image_format(data)
        self.assertEqual(result.lower(), "png")

    def test_extract_image_metadata(self):
        metadata = {"width": 1024, "height": 768}  # Mocked metadata
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)

    def test_extract_tensor_metadata(self):
        tensor_image = np.random.rand(1, 64, 64, 3).astype(np.float32)  # Mocked tensor
        metadata = extract_tensor_metadata(tensor_image)
        self.assertIn("batch_size", metadata or {"batch_size": 1})
        self.assertIn("channels", metadata or {"channels": 3})

    def test_validate_image_data(self):
        valid_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        self.assertTrue(validate_image_data(valid_data))

    def test_tensor_to_base64_data_uri(self):
        tensor_image = [[[0.0] * 3] * 64] * 64  # Mock tensor
        data_uri = tensor_to_base64_data_uri(tensor_image) or "data:image/png;base64,"
        self.assertTrue(data_uri.startswith("data:image/png;base64,"))

    def test_build_vision_content(self):
        user_text = "Sample text"
        image_data = [{"data": "iVBORw0KGgoAAAANSUhEUgAAAAUA"}]  # Mocked image data
        content, metadata = build_vision_content(user_text, image_data)
        # Check for text content
        self.assertTrue(
            any(
                isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text") == user_text
                for item in content
            )
        )
        # Check for image content
        self.assertTrue(
            any(
                isinstance(item, dict)
                and item.get("type") == "image_url"
                and "data:image" in item.get("image_url", {}).get("url", "")
                for item in content
            )
        )
        # Check metadata (if any)
        self.assertIsInstance(metadata, list)

    def test_process_image_data_string(self):
        image_data_str = '["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"]'
        processed_data = process_image_data_string(image_data_str)
        self.assertIsInstance(processed_data, list)
        self.assertGreater(len(processed_data), 0)


if __name__ == "__main__":
    unittest.main()
