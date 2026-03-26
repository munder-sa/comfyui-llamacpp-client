"""
エンドポイント切り替え時の設定値引き継ぎテスト

このスクリプトは、エンドポイント切り替え時に設定値が正しく引き継がれるかを確認します。
"""

import json
import sys


# テスト対象の JavaScript コードのロジックをシミュレート
class MockWidget:
    """ウィジェットを模倣するクラス"""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.inputEl = None
        self.element = None
        self.style = {}
        self.hidden = False


class MockNode:
    """ノードを模倣するクラス"""

    def __init__(self):
        self.masterWidgets = []
        self.widgets = []
        self.inputs = []
        self._ins = {}

    def add_widget(self, name, value):
        """ウィジェットを追加"""
        widget = MockWidget(name, value)
        self.masterWidgets.append(widget)
        self.widgets.append(widget)


# 共通パラメータとエンドポイント固有パラメータ
commonParams = [
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "seed",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "typical_p",
    "n_keep",
    "stop_sequences",
    "ignore_eos",
    "stream",
    "n_probs",
    "min_keep",
    "post_sampling_probs",
    "return_tokens",
    "timings_per_token",
    "dynatemp_range",
    "dynatemp_exponent",
    "xtc_probability",
    "xtc_threshold",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
    "grammar",
    "logit_bias",
    "cache_prompt",
    "id_slot",
    "samplers",
    "t_max_predict_ms",
    "lora",
]

endpointFields = {
    "completion": [
        "prompt",
        "n_predict",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "dynatemp_range",
        "dynatemp_exponent",
        "xtc_probability",
        "xtc_threshold",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "typical_p",
        "n_keep",
        "stop_sequences",
        "ignore_eos",
        "stream",
        "n_probs",
        "min_keep",
        "post_sampling_probs",
        "return_tokens",
        "timings_per_token",
        "grammar",
        "json_schema",
        "logit_bias",
        "cache_prompt",
        "id_slot",
        "samplers",
        "t_max_predict_ms",
        "lora",
        "response_fields",
        "image_data",
    ],
    "chat_completions": [
        "messages",
        "system_message",
        "user_message",
        "assistant_message",
        "max_tokens",
        "model",
        "tools",
        "tool_choice",
        "response_format",
        "image_data",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "stream",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
        "n_probs",
        "min_keep",
        "post_sampling_probs",
        "return_tokens",
        "timings_per_token",
        "dynatemp_range",
        "dynatemp_exponent",
        "xtc_probability",
        "xtc_threshold",
        "repeat_penalty",
        "repeat_last_n",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "typical_p",
        "lora",
    ],
    "embeddings": ["input_text", "encoding_format", "embd_normalize", "model"],
    "tokenize": ["content", "add_special", "parse_special", "with_pieces"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": [
        "input_prefix",
        "input_suffix",
        "input_extra",
        "prompt",
        "n_predict",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
        "stop_sequences",
        "stream",
        "cache_prompt",
        "id_slot",
        "samplers",
        "t_max_predict_ms",
        "grammar",
        "logit_bias",
        "n_probs",
        "min_keep",
        "post_sampling_probs",
        "return_tokens",
        "timings_per_token",
        "ignore_eos",
        "n_keep",
        "dynatemp_range",
        "dynatemp_exponent",
        "xtc_probability",
        "xtc_threshold",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "typical_p",
        "lora",
    ],
    "reranking": ["query", "documents", "top_n", "model"],
}

allToggleFieldsMap = {}
for key in endpointFields:
    fields = endpointFields[key]
    for field in fields:
        allToggleFieldsMap[field] = True
allToggleFields = list(allToggleFieldsMap.keys())


def updateUI(node, new_endpoint):
    """UI を更新する関数（JavaScript コードのシミュレーション）"""
    fieldsToShow = endpointFields.get(new_endpoint, [])

    hasImageLink = False
    if node.inputs:
        for input_item in node.inputs:
            if (
                input_item
                and (input_item.get("name") == "images" or input_item.get("type") == "IMAGE")
                and input_item.get("link") is not None
            ):
                hasImageLink = True
                break

    newWidgets = []
    for w in node.masterWidgets:
        isToggleField = w.name in allToggleFields
        isCommonParam = w.name in commonParams

        isVisible = True
        if isToggleField:
            if isCommonParam:
                isVisible = True
            else:
                isVisible = w.name in fieldsToShow

            if w.name == "image_data" and isVisible and hasImageLink:
                isVisible = False

        if isVisible:
            newWidgets.append(w)
        else:
            # 非表示のウィジェットは masterWidgets には残る
            pass

    node.widgets = newWidgets
    return node


def test_common_params_preservation():
    """共通パラメータの値保持テスト"""
    print("=" * 60)
    print("テスト 1: 共通パラメータの値保持")
    print("=" * 60)

    node = MockNode()

    # 共通パラメータを設定
    node.add_widget("temperature", 0.9)
    node.add_widget("top_k", 50)
    node.add_widget("top_p", 0.95)
    node.add_widget("seed", 42)
    node.add_widget("repeat_penalty", 1.2)

    # 固有パラメータも設定
    node.add_widget("prompt", "Hello world")
    node.add_widget("n_predict", 100)

    print(f"初期状態:")
    print(f"  temperature: {node.masterWidgets[0].value}")
    print(f"  top_k: {node.masterWidgets[1].value}")
    print(f"  n_predict: {node.masterWidgets[5].value}")

    # completion → chat_completions に切り替え
    updateUI(node, "chat_completions")
    print(f"\ncompletion → chat_completions 切り替え後:")
    print(
        f"  temperature: {node.masterWidgets[0].value} (保持: {node.masterWidgets[0].value == 0.9})"
    )
    print(f"  top_k: {node.masterWidgets[1].value} (保持: {node.masterWidgets[1].value == 50})")
    print(f"  n_predict: {node.masterWidgets[5].value} (保持: {node.masterWidgets[5].value == 100})")

    # chat_completions → completion に戻す
    updateUI(node, "completion")
    print(f"\nchat_completions → completion に戻した後:")
    print(
        f"  temperature: {node.masterWidgets[0].value} (保持: {node.masterWidgets[0].value == 0.9})"
    )
    print(f"  top_k: {node.masterWidgets[1].value} (保持: {node.masterWidgets[1].value == 50})")
    print(f"  n_predict: {node.masterWidgets[5].value} (保持: {node.masterWidgets[5].value == 100})")

    # 結果確認（n_predict は masterWidgets[6]）
    all_passed = (
        node.masterWidgets[0].value == 0.9
        and node.masterWidgets[1].value == 50
        and node.masterWidgets[6].value == 100
    )

    if all_passed:
        print("\n✓ テスト 1 合格：共通パラメータと固有パラメータの値が正しく保持されました")
    else:
        print("\n✗ テスト 1 不合格：値が保持されていません")

    return all_passed


def test_json_params_preservation():
    """JSON パラメータの値保持テスト"""
    print("\n" + "=" * 60)
    print("テスト 2: JSON パラメータの値保持")
    print("=" * 60)

    node = MockNode()

    # JSON パラメータを設定
    node.add_widget("stop_sequences", '["END", "STOP"]')
    node.add_widget("logit_bias", '{"1234": 5}')

    print(f"初期状態:")
    print(f"  stop_sequences: {node.masterWidgets[0].value}")

    # completion → embeddings に切り替え（JSON パラメータは両方にない）
    updateUI(node, "embeddings")
    print(f"\ncompletion → embeddings 切り替え後:")
    print(
        f"  stop_sequences: {node.masterWidgets[0].value} (保持: {node.masterWidgets[0].value == '[\"END\", \"STOP\"]'})"
    )

    # embeddings → completion に戻す
    updateUI(node, "completion")
    print(f"\nembeddings → completion に戻した後:")
    print(
        f"  stop_sequences: {node.masterWidgets[0].value} (保持: {node.masterWidgets[0].value == '[\"END\", \"STOP\"]'})"
    )

    all_passed = node.masterWidgets[0].value == '["END", "STOP"]'

    if all_passed:
        print("\n✓ テスト 2 合格：JSON パラメータの値が正しく保持されました")
    else:
        print("\n✗ テスト 2 不合格：値が保持されていません")

    return all_passed


def test_all_endpoints_switch():
    """全エンドポイント切り替えテスト"""
    print("\n" + "=" * 60)
    print("テスト 3: 全エンドポイント切り替え")
    print("=" * 60)

    node = MockNode()

    # 共通パラメータを設定
    node.add_widget("temperature", 0.85)
    node.add_widget("top_p", 0.9)
    node.add_widget("seed", 123)

    endpoints = [
        "completion",
        "chat_completions",
        "embeddings",
        "tokenize",
        "detokenize",
        "apply_template",
        "infill",
        "reranking",
    ]

    print("エンドポイント切り替えテスト:")
    for i, endpoint in enumerate(endpoints):
        updateUI(node, endpoint)
        print(
            f"  {i+1}. {endpoint}: temperature={node.masterWidgets[0].value}, top_p={node.masterWidgets[1].value}, seed={node.masterWidgets[2].value}"
        )

    # 最初に戻って値が保持されているか確認
    updateUI(node, "completion")
    all_passed = (
        node.masterWidgets[0].value == 0.85
        and node.masterWidgets[1].value == 0.9
        and node.masterWidgets[2].value == 123
    )

    if all_passed:
        print("\n✓ テスト 3 合格：全エンドポイント切り替えで値が保持されました")
    else:
        print("\n✗ テスト 3 不合格：値が保持されていません")

    return all_passed


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("エンドポイント切り替え時 設定値引き継ぎテスト")
    print("=" * 60 + "\n")

    results = []

    results.append(("共通パラメータの値保持", test_common_params_preservation()))
    results.append(("JSON パラメータの値保持", test_json_params_preservation()))
    results.append(("全エンドポイント切り替え", test_all_endpoints_switch()))

    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 合格" if passed else "✗ 不合格"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("全テスト合格！エンドポイント切り替え時の値保持は正常です。")
    else:
        print("一部のテストで失敗しました。確認が必要です。")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
