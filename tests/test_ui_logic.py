import unittest

from llamacpp_client_node import LlamaCppClientNode


class TestUILogic(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        self.input_types = self.node.INPUT_TYPES()

    def test_required_inputs_include_endpoint_and_server_url(self):
        required = self.input_types["required"]

        self.assertIn("server_url", required)
        self.assertIn("endpoint", required)

        self.assertEqual(required["server_url"][0], "STRING")
        self.assertEqual(required["server_url"][1]["default"], "http://127.0.0.1:8080")
        self.assertEqual(required["endpoint"][0][0], "completion")
        self.assertIn("chat_completions", required["endpoint"][0])

    def test_chat_ui_fields_exist_and_are_ordered(self):
        optional_keys = list(self.input_types["optional"].keys())

        for field in ("prompt", "system_message", "user_message", "response_format", "api_key"):
            self.assertIn(field, optional_keys)

        self.assertLess(optional_keys.index("prompt"), optional_keys.index("api_key"))
        self.assertLess(
            optional_keys.index("system_message"), optional_keys.index("response_format")
        )
        self.assertLess(optional_keys.index("user_message"), optional_keys.index("response_format"))
        self.assertLess(optional_keys.index("response_format"), optional_keys.index("api_key"))

    def test_chat_ui_fields_keep_expected_widget_configuration(self):
        optional = self.input_types["optional"]

        self.assertTrue(optional["prompt"][1]["multiline"])
        self.assertTrue(optional["system_message"][1]["multiline"])
        self.assertTrue(optional["user_message"][1]["multiline"])
        self.assertTrue(optional["assistant_message"][1]["multiline"])
        self.assertTrue(optional["messages"][1]["multiline"])
        self.assertTrue(optional["response_format"][1]["multiline"])
        self.assertFalse(optional["api_key"][1]["multiline"])

    def test_defaults_cover_values_needed_for_restore_safety(self):
        optional = self.input_types["optional"]

        self.assertEqual(optional["cache_prompt"][1]["default"], True)
        self.assertEqual(optional["timeout"][1]["default"], 600)
        self.assertEqual(optional["min_p"][1]["default"], 0.05)
        self.assertEqual(optional["top_k"][1]["default"], 40)
        self.assertEqual(optional["prompt"][1]["default"], "")


if __name__ == "__main__":
    unittest.main()
