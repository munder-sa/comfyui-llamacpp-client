import unittest

from llamacpp_client_node import LlamaCppClientNode


class TestEndpointSwitchValuePreservation(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        self.input_types = self.node.INPUT_TYPES()

    def test_common_params_preserved_on_switch(self):
        # This test documents the behavior that ALL parameters are defined in one dictionary
        # in ComfyUI, so switching 'endpoint' doesn't actually 'lose' data in the Python side,
        # but the UI might hide/show them. Here we ensure they exist in the schema.
        optional = self.input_types["optional"]
        common_fields = ["temperature", "top_k", "top_p", "min_p", "n_predict"]
        for field in common_fields:
            self.assertIn(field, optional, f"{field} should be available as an optional input")

    def test_all_endpoints_available(self):
        required = self.input_types["required"]
        endpoints = required["endpoint"][0]
        expected = [
            "completion",
            "chat_completions",
            "embeddings",
            "tokenize",
            "detokenize",
            "apply_template",
            "infill",
            "reranking",
        ]
        for ep in expected:
            self.assertIn(ep, endpoints)


class TestUIWidgetOrder(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        self.input_types = self.node.INPUT_TYPES()
        self.optional_keys = list(self.input_types["optional"].keys())

    def test_completion_priority_field_first(self):
        # 'prompt' should be near the top for usability
        self.assertIn("prompt", self.optional_keys)
        # It should appear before technical samplers like 'temperature'
        self.assertLess(self.optional_keys.index("prompt"), self.optional_keys.index("temperature"))

    def test_chat_completions_priority_fields_first(self):
        # Chat specific fields should be early
        self.assertLess(
            self.optional_keys.index("system_message"), self.optional_keys.index("temperature")
        )
        self.assertLess(
            self.optional_keys.index("user_message"), self.optional_keys.index("temperature")
        )

    def test_ui_structure_grouping(self):
        # Basic logic: Text inputs are generally before numeric samplers
        text_inputs = ["prompt", "system_message", "user_message", "assistant_message"]
        samplers = ["temperature", "top_k", "top_p"]

        for t in text_inputs:
            for s in samplers:
                if t in self.optional_keys and s in self.optional_keys:
                    self.assertLess(self.optional_keys.index(t), self.optional_keys.index(s))


class TestImageDataVisibility(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        self.input_types = self.node.INPUT_TYPES()

    def test_image_inputs_exist(self):
        optional = self.input_types["optional"]
        self.assertIn("images", optional)
        self.assertIn("image_data", optional)

    def test_multiline_config(self):
        # Ensure text areas are multiline
        optional = self.input_types["optional"]
        self.assertTrue(optional["prompt"][1].get("multiline", False))
        self.assertTrue(optional["system_message"][1].get("multiline", False))
        self.assertTrue(optional["user_message"][1].get("multiline", False))



class TestMoEModeUI(unittest.TestCase):
    """Tests for moe_mode toggle presence in INPUT_TYPES."""

    def setUp(self):
        self.node = LlamaCppClientNode()
        self.optional = self.node.INPUT_TYPES()["optional"]

    def test_moe_mode_in_optional_inputs(self):
        """moe_mode should be present as an optional BOOLEAN input."""
        self.assertIn("moe_mode", self.optional)

    def test_moe_mode_is_boolean_type(self):
        """moe_mode should be declared as BOOLEAN."""
        moe_entry = self.optional["moe_mode"]
        self.assertEqual(moe_entry[0], "BOOLEAN")

    def test_moe_mode_default_is_false(self):
        """moe_mode default must be False to avoid unintended preset activation."""
        moe_entry = self.optional["moe_mode"]
        self.assertFalse(moe_entry[1]["default"])


if __name__ == "__main__":
    unittest.main()
