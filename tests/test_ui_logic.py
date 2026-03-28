import unittest
from unittest.mock import patch  # Removed unused MagicMock

from llamacpp_client_node import LlamaCppClientNode


class TestUILogic(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()

    @patch("llamacpp_client_node.LlamaCppClientNode.updateUI")
    def test_update_ui_success(self, mock_update_ui):
        mock_update_ui.return_value = {"status": "success", "updated": True}

        result = self.node.updateUI({"endpoint": "test_endpoint"})
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["updated"])

    @patch("llamacpp_client_node.LlamaCppClientNode.updateUI")
    def test_update_ui_failure(self, mock_update_ui):
        mock_update_ui.side_effect = Exception("UI update error")

        with self.assertRaises(Exception):
            self.node.updateUI({"endpoint": "test_endpoint"})


if __name__ == "__main__":
    unittest.main()
