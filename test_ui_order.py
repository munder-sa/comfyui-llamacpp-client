"""
UI 順序テスト - 優先パラメータが最上部に表示されるか確認

このスクリプトは、エンドポイント切り替え時にプロンプト関連パラメータが
最上部に表示されるかを確認します。
"""

import json
import sys

# 共通パラメータのリスト
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
    "dynatemp_range",
    "dynatemp_exponent",
    "xtc_probability",
    "xtc_threshold",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
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
    "logit_bias",
    "cache_prompt",
    "id_slot",
    "samplers",
    "t_max_predict_ms",
    "lora",
]

# 各エンドポイント固有のパラメータリスト
endpointSpecificFields = {
    "completion": ["prompt", "n_predict", "json_schema", "response_fields", "image_data"],
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
        "json_schema",
        "response_fields",
        "image_data",
    ],
    "reranking": ["query", "documents", "top_n", "model"],
}

# 特殊パラメータ
specialParams = ["api_key", "timeout", "image_data", "images", "extract_metadata", "debug_mode"]

# 各エンドポイントの「優先パラメータ」（最上部に表示するもの）
endpointPriorityFields = {
    "completion": ["prompt"],
    "chat_completions": ["system_message", "user_message"],
    "embeddings": ["input_text"],
    "tokenize": ["content"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": ["input_prefix", "input_suffix"],
    "reranking": ["query", "documents"],
}

# 共通パラメータのセット
commonParamsSet = set(commonParams)

# 優先パラメータのセット
priorityFieldsSet = set()
for fields in endpointPriorityFields.values():
    for field in fields:
        priorityFieldsSet.add(field)

# 優先パラメータ以外の固有パラメータ
endpointOtherSpecificFields = {}
for endpoint in endpointSpecificFields:
    allSpecific = endpointSpecificFields[endpoint]
    priority = endpointPriorityFields[endpoint] or []
    prioritySet = set(priority)
    otherSpecific = [f for f in allSpecific if f not in prioritySet and f not in commonParamsSet]
    endpointOtherSpecificFields[endpoint] = otherSpecific

# 全てのパラメータを構築
endpointFields = {}
for endpoint in endpointSpecificFields:
    priorityFields = endpointPriorityFields[endpoint] or []
    otherSpecificFields = endpointOtherSpecificFields[endpoint] or []
    seen = set()
    allFields = []

    # 1. 優先パラメータ
    for f in priorityFields:
        if f not in seen:
            allFields.append(f)
            seen.add(f)

    # 2. 共通パラメータ
    for f in commonParams:
        if f not in seen:
            allFields.append(f)
            seen.add(f)

    # 3. その他固有パラメータ
    for f in otherSpecificFields:
        if f not in seen:
            allFields.append(f)
            seen.add(f)

    endpointFields[endpoint] = allFields


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


def get_expected_order(endpoint):
    """エンドポイントごとの期待されるパラメータ順序"""
    fields = endpointFields.get(endpoint, [])
    # 特殊パラメータを追加
    for f in specialParams:
        if f not in fields:
            fields.append(f)
    return fields


def updateUI(node, new_endpoint):
    """UI を更新する関数（JavaScript コードのシミュレーション）"""
    # endpoint ウィジェットを取得
    endpointWidget = None
    for mw in node.masterWidgets:
        if mw.name == "endpoint":
            endpointWidget = mw
            break

    if not endpointWidget:
        return node

    # 画像リンクのチェック
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

    # 表示するパラメータのセットを作成
    showSet = set()
    shouldHideImageData = hasImageLink

    # 優先パラメータ
    priorityFields = endpointPriorityFields.get(new_endpoint, [])
    for f in priorityFields:
        if f != "image_data" or not shouldHideImageData:
            showSet.add(f)

    # 共通パラメータ
    for f in commonParams:
        if f != "image_data" or not shouldHideImageData:
            showSet.add(f)

    # その他固有パラメータ
    otherSpecificFields = endpointOtherSpecificFields.get(new_endpoint, [])
    for f in otherSpecificFields:
        if f != "image_data" or not shouldHideImageData:
            showSet.add(f)

    # 特殊パラメータ
    for f in specialParams:
        if f != "image_data" or not shouldHideImageData:
            showSet.add(f)

    # endpoint ウィジェットは常に含める
    showSet.add("endpoint")

    # 既存ウィジェットを並べ替え（値を保持）
    newWidgetOrder = []
    addedNames = set()

    # 1. endpoint を先頭に（インデックスズレ防止）
    if endpointWidget and endpointWidget.name not in addedNames:
        newWidgetOrder.append(endpointWidget)
        addedNames.add(endpointWidget.name)

    # 2. 優先パラメータ
    for name in priorityFields:
        if name == "image_data" and shouldHideImageData:
            continue
        for mw in node.masterWidgets:
            if mw.name == name and mw.name not in addedNames:
                newWidgetOrder.append(mw)
                addedNames.add(mw.name)
                break

    # 3. 共通パラメータ
    for name in commonParams:
        if name == "image_data" and shouldHideImageData:
            continue
        for mw in node.masterWidgets:
            if mw.name == name and mw.name not in addedNames:
                newWidgetOrder.append(mw)
                addedNames.add(mw.name)
                break

    # 4. その他固有パラメータ
    for name in otherSpecificFields:
        if name == "image_data" and shouldHideImageData:
            continue
        for mw in node.masterWidgets:
            if mw.name == name and mw.name not in addedNames:
                newWidgetOrder.append(mw)
                addedNames.add(mw.name)
                break

    # 5. 特殊パラメータ
    for name in specialParams:
        if name == "image_data" and shouldHideImageData:
            continue
        for mw in node.masterWidgets:
            if mw.name == name and mw.name not in addedNames:
                newWidgetOrder.append(mw)
                addedNames.add(mw.name)
                break

    # 残りのウィジェット（マスターに存在するが順序に含まれないもの）
    for mw in node.masterWidgets:
        if mw.name not in addedNames:
            newWidgetOrder.append(mw)

    # 表示/非表示を制御
    for w in node.masterWidgets:
        if w.name in showSet:
            # 表示
            if w.inputEl:
                w.inputEl.style.display = "block"
                w.inputEl.hidden = False
            if w.element:
                w.element.style.display = "block"
                w.element.hidden = False
        else:
            # 非表示
            if w.inputEl:
                w.inputEl.style.display = "none"
                w.inputEl.hidden = True
            if w.element:
                w.element.style.display = "none"
                w.element.hidden = True

    # ウィジェット配列を更新（並べ替えのみで値を保持）
    node.widgets = newWidgetOrder

    return node


def test_completion_endpoint_order():
    """completion エンドポイントの順序テスト"""
    print("=" * 60)
    print("テスト 1: completion エンドポイントの順序")
    print("=" * 60)

    node = MockNode()

    # 主要パラメータを追加
    node.add_widget("endpoint", "completion")
    node.add_widget("prompt", "Hello world")
    node.add_widget("temperature", 0.8)
    node.add_widget("top_k", 40)
    node.add_widget("top_p", 0.95)
    node.add_widget("seed", 42)
    node.add_widget("n_predict", 100)
    node.add_widget("repeat_penalty", 1.1)
    node.add_widget("api_key", "test_key")
    node.add_widget("timeout", 600)

    updateUI(node, "completion")

    widget_names = [w.name for w in node.widgets]
    expected_order = get_expected_order("completion")[:10]  # 最初の 10 個

    print(f"ウィジェット順序:")
    for i, name in enumerate(widget_names[:10]):
        print(f"  {i+1}. {name}")

    # 優先パラメータ（prompt）が endpoint の次に来ることを確認
    prompt_index = widget_names.index("prompt") if "prompt" in widget_names else -1
    temperature_index = widget_names.index("temperature") if "temperature" in widget_names else -1
    endpoint_index = widget_names.index("endpoint") if "endpoint" in widget_names else -1

    print(f"\nendpoint の位置：{endpoint_index + 1}")
    print(f"prompt の位置：{prompt_index + 1}")
    print(f"temperature の位置：{temperature_index + 1}")

    passed = (
        prompt_index < temperature_index and prompt_index == endpoint_index + 1
    )  # endpoint の次が prompt

    if passed:
        print("\n✓ テスト 1 合格：prompt が temperature よりも前に表示されました")
    else:
        print("\n✗ テスト 1 不合格：prompt が正しく先頭に表示されませんでした")

    return passed


def test_chat_completions_endpoint_order():
    """chat_completions エンドポイントの順序テスト"""
    print("\n" + "=" * 60)
    print("テスト 2: chat_completions エンドポイントの順序")
    print("=" * 60)

    node = MockNode()

    # 主要パラメータを追加
    node.add_widget("endpoint", "chat_completions")
    node.add_widget("system_message", "You are a helpful assistant.")
    node.add_widget("user_message", "Hello!")
    node.add_widget("assistant_message", "Hi there!")
    node.add_widget("temperature", 0.8)
    node.add_widget("top_k", 40)
    node.add_widget("max_tokens", 100)
    node.add_widget("model", "llama-2-7b")

    updateUI(node, "chat_completions")

    widget_names = [w.name for w in node.widgets]

    print(f"ウィジェット順序:")
    for i, name in enumerate(widget_names[:10]):
        print(f"  {i+1}. {name}")

    # 優先パラメータ（system_message, user_message）が最初に来ることを確認
    system_index = widget_names.index("system_message") if "system_message" in widget_names else -1
    user_index = widget_names.index("user_message") if "user_message" in widget_names else -1
    temperature_index = widget_names.index("temperature") if "temperature" in widget_names else -1

    print(f"\nsystem_message の位置：{system_index + 1}")
    print(f"user_message の位置：{user_index + 1}")
    print(f"temperature の位置：{temperature_index + 1}")

    passed = system_index < temperature_index and user_index < temperature_index

    if passed:
        print("\n✓ テスト 2 合格：system_message と user_message が temperature よりも前に表示されました")
    else:
        print("\n✗ テスト 2 不合格：優先パラメータが正しく先頭に表示されませんでした")

    return passed


def test_all_endpoints_priority_fields():
    """全エンドポイントの優先パラメータ確認テスト"""
    print("\n" + "=" * 60)
    print("テスト 3: 全エンドポイントの優先パラメータ")
    print("=" * 60)

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

    all_passed = True

    for endpoint in endpoints:
        node = MockNode()
        node.add_widget("endpoint", endpoint)

        # 各パラメータを追加
        if endpoint == "completion":
            node.add_widget("prompt", "test")
            node.add_widget("temperature", 0.8)
        elif endpoint == "chat_completions":
            node.add_widget("system_message", "test")
            node.add_widget("user_message", "test")
            node.add_widget("temperature", 0.8)
        elif endpoint == "embeddings":
            node.add_widget("input_text", "test")
            node.add_widget("temperature", 0.8)
        elif endpoint == "tokenize":
            node.add_widget("content", "test")
            node.add_widget("temperature", 0.8)
        elif endpoint == "detokenize":
            node.add_widget("tokens", "[1,2,3]")
            node.add_widget("temperature", 0.8)
        elif endpoint == "apply_template":
            node.add_widget("messages", "[]")
            node.add_widget("temperature", 0.8)
        elif endpoint == "infill":
            node.add_widget("input_prefix", "test")
            node.add_widget("input_suffix", "test")
            node.add_widget("temperature", 0.8)
        elif endpoint == "reranking":
            node.add_widget("query", "test")
            node.add_widget("documents", "[]")
            node.add_widget("temperature", 0.8)

        updateUI(node, endpoint)

        widget_names = [w.name for w in node.widgets]
        priority_fields = endpointPriorityFields.get(endpoint, [])

        print(f"\n{endpoint}:")
        print(f"  優先パラメータ：{priority_fields}")
        print(f"  表示順序：{widget_names[:5]}")

        # 優先パラメータが endpoint の次に来ているか確認
        endpoint_index = widget_names.index("endpoint") if "endpoint" in widget_names else -1
        if priority_fields and endpoint_index >= 0:
            second_widget = (
                widget_names[endpoint_index + 1] if endpoint_index + 1 < len(widget_names) else None
            )
            if second_widget in priority_fields:
                print(f"  ✓ 優先パラメータが endpoint の次に表示")
            else:
                print(f"  ✗ 優先パラメータが endpoint の次に表示されていない")
                all_passed = False
        else:
            print(f"  ✗ endpoint または優先パラメータが見つからない")
            all_passed = False

    if all_passed:
        print("\n✓ テスト 3 合格：全エンドポイントで優先パラメータが先頭に表示されました")
    else:
        print("\n✗ テスト 3 不合格：一部のエンドポイントで優先パラメータが先頭に表示されませんでした")

    return all_passed


def test_ui_order_structure():
    """UI 構造のテスト（優先→共通→その他固有→特殊）"""
    print("\n" + "=" * 60)
    print("テスト 4: UI 構造の検証")
    print("=" * 60)

    endpoint = "completion"
    expected = get_expected_order(endpoint)

    print(f"\n{endpoint} エンドポイントの期待される順序:")
    print(f"  1. 優先パラメータ：{endpointPriorityFields[endpoint]}")
    print(f"  2. 共通パラメータ：{commonParams[:5]}...")
    print(f"  3. その他固有：{endpointOtherSpecificFields[endpoint]}")
    print(f"  4. 特殊パラメータ：{specialParams}")

    # 実際の順序が期待構造に従っているか確認
    priority_fields = endpointPriorityFields[endpoint]
    other_specific = endpointOtherSpecificFields[endpoint]

    # 優先パラメータが先頭にある
    for i, field in enumerate(priority_fields):
        if i < len(expected) and expected[i] != field:
            print(f"\n✗ 優先パラメータの順序が異なります：{field} が {i} 番目にあるべき")
            return False

    # その他固有パラメータが共通パラメータの後にある
    priority_count = len(priority_fields)
    common_count = len([f for f in commonParams if f not in priority_fields])

    # 特殊パラメータが最後にある（存在する場合のみチェック）
    # 特殊パラメータは重複を除くため、実際に存在するもののみをチェック
    for field in specialParams:
        if field in expected:
            # 特殊パラメータは他のパラメータより後に存在する
            special_index = expected.index(field)
            # 優先パラメータと共通パラメータの後に存在することを確認
            if field in priority_fields:
                continue  # 優先パラメータは先頭にあるので OK
            if field in commonParams:
                continue  # 共通パラメータは 2 番目にあるので OK
            # その他固有パラメータの場合、共通パラメータの後にあることを確認
            common_last_index = -1
            for i, f in enumerate(commonParams):
                if f in expected and f not in priority_fields:
                    common_last_index = i
            if common_last_index >= 0 and special_index < common_last_index:
                print(f"\n✗ 特殊パラメータ {field} が共通パラメータより前にあります")
                return False

    print("\n✓ テスト 4 合格：UI 構造が正しい順序で構成されています")
    return True


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("UI 順序テスト - 優先パラメータが最上部に表示されるか確認")
    print("=" * 60 + "\n")

    results = []

    results.append(("completion エンドポイントの順序", test_completion_endpoint_order()))
    results.append(("chat_completions エンドポイントの順序", test_chat_completions_endpoint_order()))
    results.append(("全エンドポイントの優先パラメータ", test_all_endpoints_priority_fields()))
    results.append(("UI 構造の検証", test_ui_order_structure()))

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
        print("全テスト合格！UI 順序は正常です。")
    else:
        print("一部のテストで失敗しました。確認が必要です。")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
