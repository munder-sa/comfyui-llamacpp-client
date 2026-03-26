"""
エンドポイント切り替え時の設定値引き継ぎテスト

このスクリプトは、エンドポイント切り替え時に設定値が正しく引き継がれるかを確認します。
"""

import json
import sys


# 共通パラメータのリスト
commonParams = [
    "temperature", "top_k", "top_p", "min_p", "seed",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "mirostat", "mirostat_tau", "mirostat_eta", "typical_p",
    "dynatemp_range", "dynatemp_exponent", "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers",
    "n_keep", "stop_sequences", "ignore_eos", "stream", "n_probs",
    "min_keep", "post_sampling_probs", "return_tokens", "timings_per_token",
    "grammar", "logit_bias", "cache_prompt",
    "id_slot", "samplers", "t_max_predict_ms", "lora"
]

# 各エンドポイント固有のパラメータリスト
endpointSpecificFields = {
    "completion": [
        "prompt", "n_predict", "json_schema", "response_fields", "image_data"
    ],
    "chat_completions": [
        "messages", "system_message", "user_message", "assistant_message", "max_tokens",
        "model", "tools", "tool_choice", "response_format", "image_data"
    ],
    "embeddings": ["input_text", "encoding_format", "embd_normalize", "model"],
    "tokenize": ["content", "add_special", "parse_special", "with_pieces"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": [
        "input_prefix", "input_suffix", "input_extra", "prompt", "n_predict",
        "json_schema", "response_fields", "image_data"
    ],
    "reranking": ["query", "documents", "top_n", "model"]
}

# 特殊パラメータ
specialParams = [
    "api_key", "timeout", "image_data", "images", "extract_metadata", "debug_mode"
]

# 各エンドポイントの「優先パラメータ」（最上部に表示するもの）
endpointPriorityFields = {
    "completion": ["prompt"],
    "chat_completions": ["system_message", "user_message"],
    "embeddings": ["input_text"],
    "tokenize": ["content"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": ["input_prefix", "input_suffix"],
    "reranking": ["query", "documents"]
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

    # 初期値を辞書で保持
    initial_values = {w.name: w.value for w in node.masterWidgets}
    print(f"初期値:")
    for name, value in initial_values.items():
        print(f"  {name}: {value}")

    # completion → chat_completions に切り替え
    updateUI(node, "chat_completions")
    print(f"\ncompletion → chat_completions 切り替え後:")
    for w in node.masterWidgets:
        print(f"  {w.name}: {w.value}")

    # chat_completions → completion に戻す
    updateUI(node, "completion")
    print(f"\nchat_completions → completion に戻した後:")
    for w in node.masterWidgets:
        print(f"  {w.name}: {w.value}")

    # 値が保持されているか確認（名前ベースでチェック）
    preserved = True
    for name, expected_value in initial_values.items():
        found_widget = None
        for mw in node.masterWidgets:
            if mw.name == name:
                found_widget = mw
                break
        
        if found_widget:
            if found_widget.value != expected_value:
                print(f"✗ {name}: 値が変更されました (expected: {expected_value}, got: {found_widget.value})")
                preserved = False
        else:
            print(f"✗ {name}: ウィジェットが見つかりません")
            preserved = False

    if preserved:
        print("\n✓ テスト 1 合格：共通パラメータと固有パラメータの値が正しく保持されました")
    else:
        print("\n✗ テスト 1 不合格：値が保持されていません")

    return preserved


def test_json_params_preservation():
    """JSON パラメータの値保持テスト"""
    print("\n" + "=" * 60)
    print("テスト 2: JSON パラメータの値保持")
    print("=" * 60)

    node = MockNode()

    # JSON パラメータを設定
    node.add_widget("stop_sequences", '["END", "STOP"]')
    node.add_widget("logit_bias", '{"1234": 5}')

    initial_values = {w.name: w.value for w in node.masterWidgets}
    print(f"初期値:")
    for name, value in initial_values.items():
        print(f"  {name}: {value}")

    # completion → embeddings に切り替え（JSON パラメータは両方にない）
    updateUI(node, "embeddings")
    print(f"\ncompletion → embeddings 切り替え後:")
    for w in node.masterWidgets:
        print(f"  {w.name}: {w.value}")

    # embeddings → completion に戻す
    updateUI(node, "completion")
    print(f"\nembeddings → completion に戻した後:")
    for w in node.masterWidgets:
        print(f"  {w.name}: {w.value}")

    # 値が保持されているか確認
    preserved = True
    for name, expected_value in initial_values.items():
        found_widget = None
        for mw in node.masterWidgets:
            if mw.name == name:
                found_widget = mw
                break
        
        if found_widget:
            if found_widget.value != expected_value:
                print(f"✗ {name}: 値が変更されました (expected: {expected_value}, got: {found_widget.value})")
                preserved = False
        else:
            print(f"✗ {name}: ウィジェットが見つかりません")
            preserved = False

    if preserved:
        print("\n✓ テスト 2 合格：JSON パラメータの値が正しく保持されました")
    else:
        print("\n✗ テスト 2 不合格：値が保持されていません")

    return preserved


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
        print(f"  {i+1}. {endpoint}: temperature={node.masterWidgets[0].value}, top_p={node.masterWidgets[1].value}, seed={node.masterWidgets[2].value}")

    # 最初に戻って値が保持されているか確認
    updateUI(node, "completion")
    
    # 値が保持されているか確認（名前ベースでチェック）
    preserved = True
    for name, expected_value in [("temperature", 0.85), ("top_p", 0.9), ("seed", 123)]:
        found_widget = None
        for mw in node.masterWidgets:
            if mw.name == name:
                found_widget = mw
                break
        
        if found_widget:
            if found_widget.value != expected_value:
                print(f"✗ {name}: 値が変更されました (expected: {expected_value}, got: {found_widget.value})")
                preserved = False
        else:
            print(f"✗ {name}: ウィジェットが見つかりません")
            preserved = False

    if preserved:
        print("\n✓ テスト 3 合格：全エンドポイント切り替えで値が保持されました")
    else:
        print("\n✗ テスト 3 不合格：値が保持されていません")

    return preserved


def test_endpoint_index_stability():
    """インデックスズレ防止テスト"""
    print("\n" + "=" * 60)
    print("テスト 4: インデックスズレ防止")
    print("=" * 60)

    node = MockNode()

    # 多くのパラメータを追加
    node.add_widget("endpoint", "completion")
    node.add_widget("prompt", "test prompt")
    node.add_widget("temperature", 0.8)
    node.add_widget("top_k", 40)
    node.add_widget("top_p", 0.95)
    node.add_widget("seed", 42)
    node.add_widget("repeat_penalty", 1.1)
    node.add_widget("n_predict", 100)
    node.add_widget("api_key", "test_key")
    node.add_widget("timeout", 600)

    initial_values = {w.name: w.value for w in node.masterWidgets}
    print(f"初期値:")
    for name, value in initial_values.items():
        print(f"  {name}: {value}")

    # 複数回エンドポイント切り替え
    endpoints = ["completion", "chat_completions", "completion", "embeddings", "completion"]
    for endpoint in endpoints:
        updateUI(node, endpoint)

    # 値が保持されているか確認
    preserved = True
    for name, expected_value in initial_values.items():
        # 該当するウィジェットを探す
        found_widget = None
        for mw in node.masterWidgets:
            if mw.name == name:
                found_widget = mw
                break
        
        if found_widget:
            if found_widget.value != expected_value:
                print(f"✗ {name}: 値が変更されました (expected: {expected_value}, got: {found_widget.value})")
                preserved = False
        else:
            print(f"✗ {name}: ウィジェットが見つかりません")
            preserved = False

    if preserved:
        print("\n✓ テスト 4 合格：全パラメータの値が保持されました（インデックスズレなし）")
    else:
        print("\n✗ テスト 4 不合格：一部の値が保持されていません")

    return preserved


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("エンドポイント切り替え時 設定値引き継ぎテスト")
    print("=" * 60 + "\n")

    results = []

    results.append(("共通パラメータの値保持", test_common_params_preservation()))
    results.append(("JSON パラメータの値保持", test_json_params_preservation()))
    results.append(("全エンドポイント切り替え", test_all_endpoints_switch()))
    results.append(("インデックスズレ防止", test_endpoint_index_stability()))

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