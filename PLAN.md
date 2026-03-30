# comfyui-llamacpp-client プロジェクト計画

最終更新: 2026-03-30

---

## プロジェクト現状サマリー

### 完了済みフェーズ

| フェーズ | 内容 | 状態 |
|---|---|---|
| コア実装 | `llamacpp_client_node.py` / `utils/` 全モジュール | ✅ 完了 |
| テスト基盤構築 | `tests/` ディレクトリ再編、`tests/__init__.py` / `helpers.py` 作成 | ✅ 完了 |
| test_param_utils | `param_utils.py` 全関数テスト (109テスト中の一部) | ✅ 完了 |
| test_llama_client | 全エンドポイント + リトライ + レスポンス検証テスト (491行, 充実) | ✅ 完了 |
| 旧テストファイル整理 | ルートレベルの旧テストファイルを削除 | ✅ 完了 |
| pyproject.toml 設定 | pytest / dev dependencies 設定追加 | ✅ 完了 |
| メッセージ構築リファクタ | `chat_completions` のシステムメッセージ先頭配置保証 | ✅ 完了 |

### 現在のテスト状況

```
109 passed, 2 warnings  (2026-03-30 時点)
```

venv: `D:\AI\ComfyUI\venv\Scripts\python.exe`

```bash
# 全テスト実行
D:\AI\ComfyUI\venv\Scripts\python.exe -m pytest tests/ -v --no-cov

# カバレッジ付き
D:\AI\ComfyUI\venv\Scripts\python.exe -m pytest tests/ -v --cov=utils --cov=llamacpp_client_node --cov-report=term-missing
```

---

## フェーズ1: テスト品質向上（進行中）

現在の3ファイルは「簡略版」実装のため、旧PLAN.mdの詳細仕様に準拠させる。

### 1-1. `tests/test_image_utils.py` のリライト

**現状**: 1クラス `TestImageUtils` に7メソッド（全般的なスモークテスト）
**目標**: 5クラスに分割し、エッジケースを網羅

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

### 1-2. `tests/test_node.py` のリライト

**現状**: 1クラス `TestNode` に13メソッド（`_extract_response_text` 中心、`process_request` 統合テストなし）
**目標**: 4クラスに分割し、`process_request` の全エンドポイント分岐をモックでテスト

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

### 1-3. `tests/test_ui_logic.py` の拡充

**現状**: 1クラス `TestUILogic` に4メソッド（INPUT_TYPES の構造確認のみ）
**目標**: 3クラスに分割し、エンドポイント切り替えロジックと Widget 順序を検証

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

### 実装順序

1. `[ ]` `tests/test_image_utils.py` をリライト（5クラス、19メソッド以上）
2. `[ ]` `tests/test_node.py` をリライト（4クラス、16メソッド以上）
3. `[ ]` `tests/test_ui_logic.py` を拡充（3クラス、10メソッド以上）
4. `[ ]` 全テストが通ることを確認（`pytest tests/ -v --no-cov`）

---

## フェーズ2: CI/CD 整備

### 2-1. GitHub Actions ワークフロー構築

`.github/workflows/` に以下のワークフローを作成する。

#### `ci.yml` — Pull Request・Push 時の自動テスト

```yaml
# トリガー: push to main/develop, PR to main
# ジョブ:
#   1. lint (flake8, mypy)
#   2. test (pytest --cov, カバレッジレポート)
# Python バージョン: 3.10, 3.11 (matrix)
```

**実装項目:**
- `[ ]` `.github/workflows/ci.yml` を作成
  - `actions/checkout@v4`
  - `actions/setup-python@v5` (matrix: 3.10, 3.11)
  - pip install `.[dev]`
  - `flake8` lint チェック
  - `mypy` 型チェック（`mypy.ini` 参照）
  - `pytest tests/ --cov=utils --cov=llamacpp_client_node --cov-report=xml`
  - Codecov へのカバレッジアップロード（オプション）

#### `release.yml` — タグプッシュ時のリリース自動化

```yaml
# トリガー: push tag v*.*.*
# ジョブ:
#   1. テスト実行（ci.yml 再利用 or inline）
#   2. CHANGELOG.md から リリースノート抽出
#   3. GitHub Release 作成
```

**実装項目:**
- `[ ]` `.github/workflows/release.yml` を作成
  - `v*.*.*` タグトリガー
  - テスト合格を条件とする
  - `CHANGELOG.md` 内の最新バージョンエントリを抽出してリリースノートに使用
  - `actions/create-release@v1` または `softprops/action-gh-release` でリリース作成

### 2-2. 品質チェック強化

- `[ ]` `pyproject.toml` の `[tool.pytest.ini_options]` を修正
  - `testpaths = ["tests"]` を追加
  - `addopts` を `"-v --tb=short"` に整理（カバレッジはCI専用に切り分け）
- `[ ]` `.pre-commit-config.yaml` の動作確認と必要に応じた更新
- `[ ]` `mypy` がクリーンにパスすることを確認

### 2-3. バッジ追加

- `[ ]` `README.md` にバッジを追加
  - CI ステータスバッジ（GitHub Actions）
  - テストカバレッジバッジ（Codecov）
  - Python バージョンバッジ

---

## 技術的制約・注意事項

### 実行環境

- **Python venv**: `D:\AI\ComfyUI\venv\Scripts\python.exe`
- システム Python は**使用禁止**
- GitHub Actions では `python-version: ["3.10", "3.11"]` matrix を使用

### テスト方針

- **全テストはサーバー不要（Mockベース）**
- `unittest.TestCase` ベースで統一
- `pytest` で一括実行可能
- IMAGE テンソル: `[Batch, Height, Width, Channels]` float32, 0-1 range

### ComfyUI 固有仕様

- `process_request()` は **5-tuple** を返す: `(response_text, raw_response, error, status_code, metadata)`
- `handle_chat_completions()` は `ApiResponse` を返す（`.unpack()` で 5-tuple に展開）
- エンドポイント一覧: `completion`, `chat_completions`, `embeddings`, `tokenize`, `detokenize`, `apply_template`, `infill`, `reranking`

---

## 実装履歴参照

詳細な変更履歴は `REVISION_HISTORY.md` を参照。

| 日付 | 主な変更 |
|---|---|
| 2026-03-22 | システムメッセージ配置ロジック改善・メタデータ抽出強化 |
| 2026-03-21 | デバッグログ追加・パラメータ処理改善 |
| 2026-03-21 | 実行ロジック復元・マルチモーダル機能統合・JSON パース徹底 |
