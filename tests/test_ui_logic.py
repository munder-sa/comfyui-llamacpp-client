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

    def test_json_params_preserved_on_switch(self):
        optional = self.input_types["optional"]
        json_fields = ["samplers", "stop_sequences", "logit_bias", "messages", "tools"]
        for field in json_fields:
            self.assertIn(field, optional)
            self.assertEqual(optional[field][0], "STRING")

    def test_all_endpoints_round_trip(self):
        # The node class in Python does not dynamically remove inputs;
        # Instead, we just verify `INPUT_TYPES` returns the full dictionary unconditionally.
        optional1 = self.node.INPUT_TYPES()["optional"]
        optional2 = self.node.INPUT_TYPES()["optional"]
        self.assertEqual(list(optional1.keys()), list(optional2.keys()))

    def test_index_stability_multiple_switches(self):
        # Verify that multiple consecutive requests for INPUT_TYPES() yield identical order
        keys_run1 = list(self.node.INPUT_TYPES()["optional"].keys())
        keys_run2 = list(self.node.INPUT_TYPES()["optional"].keys())
        keys_run3 = list(self.node.INPUT_TYPES()["optional"].keys())
        self.assertEqual(keys_run1, keys_run2)
        self.assertEqual(keys_run2, keys_run3)


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

    def test_all_endpoints_priority_field_after_endpoint(self):
        # Priority fields in JS are defined for each endpoint
        # to sit immediately after 'endpoint' / 'server_url'.
        # Since 'endpoint' and 'server_url' are required, they appear at the top visually.
        # So we verify that all JS priority fields appear *before* standard fields.
        js_priority_fields = [
            "prompt",
            "system_message",
            "user_message",
            "input_text",
            "content",
            "tokens",
            "messages",
            "input_prefix",
            "input_suffix",
            "query",
            "documents",
        ]
        temp_idx = self.optional_keys.index("temperature")
        for f in js_priority_fields:
            if f in self.optional_keys:
                self.assertLess(
                    self.optional_keys.index(f),
                    temp_idx,
                    f"Priority field {f} should be before temperature",
                )

    def test_ui_structure_order(self):
        # Tests priority -> ... -> image_data -> common(temperature) -> dry(dry_multiplier)
        # Python defines them in groups: endpoint_specific -> sampling -> dry
        optional_keys = self.optional_keys
        if all(
            k in optional_keys for k in ["prompt", "image_data", "temperature", "dry_multiplier"]
        ):
            valid_order = (
                optional_keys.index("prompt")
                < optional_keys.index("image_data")
                < optional_keys.index("temperature")
                < optional_keys.index("dry_multiplier")
            )
            self.assertTrue(valid_order)


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

    def test_image_data_hidden_when_image_link(self):
        import os
        import re

        js_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "web", "llamacpp_client_extension.js"
        )
        if not os.path.exists(js_path):
            # If path structure is different, adjust or skip
            js_path = os.path.join(
                os.path.dirname(__file__), "..", "web", "llamacpp_client_extension.js"
            )
            if not os.path.exists(js_path):
                self.skipTest("JS file not found, skipping static analysis.")

        with open(js_path, "r", encoding="utf-8") as f:
            js_code = f.read()

        # Verify JS has the logic string "shouldHideImageData = hasImageLink;"
        self.assertIn("shouldHideImageData = hasImageLink", js_code)

        # Verify it tries to skip image_data if shouldHideImageData
        self.assertTrue(
            re.search(r'f !== "image_data" \|\| !shouldHideImageData', js_code),
            "JS does not contain the logic to hide image_data conditionally",
        )

    def test_image_data_shown_when_no_image_link(self):
        # Tested symmetrically through the JS regex string search above.
        # Also ensure Python distinguishes the types correctly: images=IMAGE, image_data=STRING
        optional = self.input_types["optional"]
        self.assertEqual(optional["images"][0], "IMAGE")
        self.assertEqual(optional["image_data"][0], "STRING")


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
