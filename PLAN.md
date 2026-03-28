# テスト整理・品質向上 実装計画

## 概要

現状の散在したテストファイルを整理し、`unittest.TestCase` ベースで統一された
テストスイートに再構築する。全テストはサーバー不要（Mock ベース）で動作し、
`pytest` で一括実行可能にする。

---

## 現状の問題点

### バグ・欠陥

| ファイル | 行 | 問題 |
|---|---|---|
| `test_node.py` | 34, 59, 95, 124 | `process_request()` は `5-tuple` を返すが `4-tuple` でアンパック |
| `test_optimized_features.py` | 147 | `MockLlamaClient._make_request` の型ヒントが `Tuple[str,str,str,int]`（正: `Tuple[dict,str,str,int]`） |
| `test_optimized_features.py` | 51-71 | `MockLlamaServer.create_completion_response` が `{"choices":[...]}` を返すが `/completion` は `{"content":"..."}` が正しい形式 |
| `test_endpoint_switch.py` / `test_ui_order.py` | 全体 | `MockWidget` / `MockNode` / `updateUI` が完全重複 |

### カバレッジの欠如

- `utils/param_utils.py`: `map_parameters()` / `safe_convert_to_int()` / `safe_convert_to_float()` が**未テスト**
- `utils/image_utils.py`: `detect_image_format()` / `validate_image_data()` / `process_image_data_string()` が**未テスト**
- `utils/llama_client.py`: `handle_embeddings()` / `handle_tokenize()` / `handle_detokenize()` / `handle_apply_template()` / `handle_infill()` / `handle_reranking()` が**未テスト**
- `llamacpp_client_node.py`: `process_request()` の全エンドポイント分岐が**未テスト**

### 構造上の問題

- `test_endpoint_switch.py` / `test_ui_order.py` / `test_fix.py`: `unittest` 未使用・印刷ベース→ CI 統合不可
- `test_node.py`: 実サーバーへの接続を前提とした統合テスト→ CI 不可

---

## 新しいディレクトリ構造

```
comfyui-llamacpp-client/
├── tests/
│   ├── __init__.py
│   ├── helpers.py               # 共通 Mock / ヘルパー
│   ├── test_param_utils.py      # NEW: param_utils 全関数テスト
│   ├── test_image_utils.py      # NEW: image_utils 全関数テスト
│   ├── test_llama_client.py     # REWRITE: 全エンドポイント + エラーハンドリング
│   ├── test_node.py             # REWRITE: LlamaCppClientNode.process_request 全分岐 (Mock)
│   └── test_ui_logic.py         # MERGE: endpoint_switch + ui_order を統合・unittest 化
│
│   # 以下は移行完了後に削除
├── test_node.py                 → tests/test_node.py に移行後、削除
├── test_error_handling.py       → tests/test_llama_client.py に統合後、削除
├── test_optimized_features.py   → tests/test_image_utils.py + test_llama_client.py に分配後、削除
├── test_endpoint_switch.py      → tests/test_ui_logic.py に統合後、削除
├── test_ui_order.py             → tests/test_ui_logic.py に統合後、削除
└── test_fix.py                  → tests/test_param_utils.py に統合後、削除
```

---

## 各ファイルの詳細仕様

### `tests/helpers.py`

```python
# 提供するもの:
# - _make_mock_response(status_code, json_data, text, raise_json) -> MagicMock
# - SessionPatchMixin: _patch_session_post(client, side_effect, return_value)
# - MockWidget(name, value)
# - MockNode: add_widget(name, value) / masterWidgets / widgets / inputs
# - make_completion_response(content) -> dict  # {"content": content}
# - make_chat_response(content) -> dict        # {"choices":[{"message":{"content":content}}]}
# - create_numpy_image(h, w, channels, value) -> np.ndarray  # float32, 0-1 range
```

---

### `tests/test_param_utils.py` (新規)

| テストクラス | テストメソッド | 検証内容 |
|---|---|---|
| `TestCleanParams` | `test_none_values_removed` | `None` 値が削除される |
| | `test_empty_string_removed` | 空文字が削除される |
| | `test_json_string_parsed` | `stop_sequences`等 JSON 文字列がパースされる |
| | `test_list_passthrough` | 既に list の場合はそのまま通過 |
| | `test_invalid_json_fallback_to_empty_list` | 不正 JSON は `[]` にフォールバック |
| | `test_image_data_newline_stripped` | `image_data` の改行が除去される |
| `TestMapParameters` | `test_basic_mapping` | mapping 通りにキーが変換される |
| | `test_missing_keys_skipped` | mapping にないキーはスキップ |
| | `test_none_values_skipped` | `None` 値はスキップ |
| `TestSafeConvertToInt` | `test_normal_conversion` | 通常の int 変換 |
| | `test_float_string_conversion` | `"1.0"` → `1` |
| | `test_invalid_string_returns_default` | 不正文字列はデフォルト値を返す |
| | `test_min_max_clipping` | min/max 範囲外はデフォルト値を返す |
| `TestSafeConvertToFloat` | `test_normal_conversion` | 通常の float 変換 |
| | `test_randomize_returns_default` | `"randomize"` はデフォルト値を返す |
| | `test_min_max_clipping` | min/max 範囲外はデフォルト値を返す |

---

### `tests/test_image_utils.py` (新規)

| テストクラス | テストメソッド | 検証内容 |
|---|---|---|
| `TestDetectImageFormat` | `test_jpeg_magic_bytes` | JPEG マジックナンバー検出 |
| | `test_png_magic_bytes` | PNG マジックナンバー検出 |
| | `test_invalid_data_returns_none` | 不正データは `None` を返す |
| | `test_base64_string_input` | Base64 文字列入力でも検出できる |
| `TestValidateImageData` | `test_valid_base64` | 有効な Base64 は `True` |
| | `test_data_uri_format` | `data:image/...` 形式は `True` |
| | `test_empty_string_invalid` | 空文字は `False` |
| `TestProcessImageDataString` | `test_valid_json_array` | 有効な JSON 配列をパース |
| | `test_empty_string_returns_empty_list` | 空文字は `[]` を返す |
| | `test_invalid_json_returns_empty_list` | 不正 JSON は `[]` を返す |
| `TestTensorToBase64DataUri` | `test_numpy_array_conversion` | numpy 配列 → Base64 Data URI |
| | `test_jpeg_quality_affects_size` | quality が出力サイズに影響する |
| | `test_large_image_resized` | `max_dimension` で縮小される |
| | `test_png_output_format` | PNG フォーマット出力 |
| | `test_invalid_input_returns_none` | 不正入力は `None` を返す |
| `TestBuildVisionContent` | `test_text_only` | テキストのみの場合 |
| | `test_text_and_image_data` | テキスト + `image_data` JSON |
| | `test_text_and_tensor_images_4d` | テキスト + 4D ComfyUI テンソル `[B,H,W,C]` |
| | `test_metadata_extraction` | `extract_metadata=True` でメタデータ取得 |
| | `test_empty_inputs` | 空入力は空リストを返す |

---

### `tests/test_llama_client.py` (既存 `test_error_handling.py` を拡張・移行)

| テストクラス | テストメソッド | 検証内容 |
|---|---|---|
| `TestMakeRequest` | `test_successful_completion_response` | `content` キーある場合に正常 200 |
| | `test_successful_chat_response` | `choices` キーある場合に正常 200 |
| | `test_http_400_error` | 400 → error に "400" が含まれる |
| | `test_http_500_error` | 500 → error に "500" が含まれる |
| | `test_invalid_json_response` | JSONDecodeerror → 502 / "Invalid JSON" |
| | `test_invalid_structure_returns_error` | `content`/`text`/`choices` なし → "Invalid structure" |
| | `test_think_tag_stripped` | `<think>...</think>` が除去される |
| | `test_prompt_tag_extracted` | `<prompt>...</prompt>` の中身が抽出される |
| `TestRetryDecorator` | `test_connection_error_retried_3_times` | 3 回リトライ後に re-raise |
| | `test_timeout_error_retried_3_times` | 3 回リトライ後に re-raise |
| | `test_success_on_second_attempt` | 2 回目で成功 → call_count == 2 |
| | `test_non_transient_error_not_retried` | `RequestException` はリトライしない → 500 |
| `TestHandleCompletion` | `test_basic_completion` | 正常 completion レスポンス |
| | `test_stop_sequences_forced` | `params["stop"]` が強制上書きされる |
| | `test_http_error` | 400 → error あり |
| `TestHandleChatCompletions` | `test_system_and_user_message` | system + user がメッセージに組み込まれる |
| | `test_messages_history_prepended` | `messages` 履歴が正しく前置される |
| | `test_assistant_prefill` | `assistant_message` がメッセージに追加される |
| | `test_image_data_included` | `image_data` が vision content に含まれる |
| | `test_tensor_images_included` | `images` テンソルが vision content に含まれる |
| | `test_returns_5_tuple` | 戻り値は 5-tuple |
| | `test_error_returns_5_tuple` | エラー時も 5-tuple |
| `TestHandleEmbeddings` | `test_basic_embeddings` | embeddings エンドポイント正常系 |
| | `test_http_error` | 400 → error あり |
| `TestHandleTokenize` | `test_basic_tokenize` | tokenize エンドポイント正常系 |
| `TestHandleDetokenize` | `test_basic_detokenize` | detokenize エンドポイント正常系 |
| `TestHandleApplyTemplate` | `test_basic_apply_template` | apply_template エンドポイント正常系 |
| `TestHandleInfill` | `test_basic_infill` | infill エンドポイント正常系 |
| `TestHandleReranking` | `test_basic_reranking` | reranking エンドポイント正常系 |

---

### `tests/test_node.py` (既存 `test_node.py` を全面 rewrite)

| テストクラス | テストメソッド | 検証内容 |
|---|---|---|
| `TestLlamaCppClientNodeStructure` | `test_input_types_has_required_keys` | `server_url` / `endpoint` が required に存在 |
| | `test_return_types_5_tuple` | `RETURN_TYPES` が 5 要素 |
| | `test_return_names_5_tuple` | `RETURN_NAMES` が 5 要素 |
| | `test_node_category` | `CATEGORY == "AI/LlamaCpp"` |
| `TestProcessRequestCompletion` | `test_completion_success` | completion 正常系 → response_text が抽出される |
| | `test_completion_error_status` | 400 → error が返る |
| | `test_completion_exception` | 例外 → status_code 500 |
| `TestProcessRequestChatCompletions` | `test_chat_completions_success` | chat 正常系 → choices[0] content が response_text |
| | `test_chat_completions_with_images` | `images` テンソルが渡せる |
| `TestProcessRequestOtherEndpoints` | `test_embeddings_endpoint` | embeddings 分岐が呼ばれる |
| | `test_tokenize_endpoint` | tokenize 分岐が呼ばれる |
| | `test_detokenize_endpoint` | detokenize 分岐が呼ばれる |
| | `test_apply_template_endpoint` | apply_template 分岐が呼ばれる |
| | `test_infill_endpoint` | infill 分岐が呼ばれる |
| | `test_reranking_endpoint` | reranking 分岐が呼ばれる |
| `TestProcessRequestMetadata` | `test_metadata_populated_from_chat` | chat で metadata_list があれば metadata dict に収録される |

---

### `tests/test_ui_logic.py` (新規: `test_endpoint_switch.py` + `test_ui_order.py` を統合)

| テストクラス | テストメソッド | 検証内容 |
|---|---|---|
| `TestEndpointSwitchValuePreservation` | `test_common_params_preserved_on_switch` | 共通パラメータの値が endpoint 切り替えで保持される |
| | `test_json_params_preserved_on_switch` | JSON パラメータの値が保持される |
| | `test_all_endpoints_round_trip` | 全エンドポイントを順番に切り替えて値が保持される |
| | `test_index_stability_multiple_switches` | 複数回切り替えでインデックスズレなし |
| `TestUIWidgetOrder` | `test_completion_priority_field_first` | `prompt` が `temperature` より前に表示 |
| | `test_chat_completions_priority_fields_first` | `system_message` / `user_message` が `temperature` より前 |
| | `test_all_endpoints_priority_field_after_endpoint` | 全エンドポイントで優先フィールドが `endpoint` の直後 |
| | `test_ui_structure_order` | 優先→共通→その他固有→特殊 の順序を検証 |
| `TestImageDataVisibility` | `test_image_data_hidden_when_image_link` | `images` リンクあり → `image_data` が非表示 |
| | `test_image_data_shown_when_no_image_link` | `images` リンクなし → `image_data` が表示 |

---

## `pyproject.toml` への追加設定

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

---

## 既存ファイルの処置方針

| 現ファイル | 処置 |
|---|---|
| `test_node.py` | `tests/test_node.py` に rewrite 後、ルート版は削除 |
| `test_error_handling.py` | `tests/test_llama_client.py` に統合後、削除 |
| `test_optimized_features.py` | `tests/test_image_utils.py` + `tests/test_llama_client.py` に分配後、削除 |
| `test_endpoint_switch.py` | `tests/test_ui_logic.py` に統合後、削除 |
| `test_ui_order.py` | `tests/test_ui_logic.py` に統合後、削除 |
| `test_fix.py` | `tests/test_param_utils.py` に統合後、削除 |

---

## 実装順序

1. `tests/__init__.py` を作成
2. `tests/helpers.py` を作成（共通ヘルパー）
3. `tests/test_param_utils.py` を作成
4. `tests/test_image_utils.py` を作成
5. `tests/test_llama_client.py` を作成（`test_error_handling.py` の内容を移行・拡充）
6. `tests/test_node.py` を作成（`test_node.py` を全面 rewrite）
7. `tests/test_ui_logic.py` を作成（`test_endpoint_switch.py` + `test_ui_order.py` を統合）
8. `pyproject.toml` に pytest 設定を追加
9. ルートレベルの旧テストファイルを削除

---

## 実行コマンド（実装後）

```bash
# 全テスト実行
D:\AI\ComfyUI\venv\Scripts\python.exe -m pytest tests/ -v

# カバレッジレポート付き（pytest-cov が必要）
D:\AI\ComfyUI\venv\Scripts\python.exe -m pytest tests/ -v --cov=utils --cov=llamacpp_client_node --cov-report=term-missing
```
