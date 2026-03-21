# LlamaCpp Client Node 修正履歴

## 2026-03-22: システムメッセージ配置ロジックの改善とデバッグ機能の強化

### 変更内容

#### 1. システムメッセージの信頼性の高い配置 (`llamacpp_client_node.py`)
- `chat_completions` エンドポイントのメッセージ構築ロジックを完全リファクタリング
- スタックベースのアプローチでシステムメッセージを常に先頭に配置することを保証
- 以下の優先順位でメッセージを構築：
  1. システムプロンプト（最優先）
  2. 履歴メッセージ（重複防止）
  3. ユーザーメッセージとビジョンコンテンツの統合
  4. アシスタントメッセージ

```python
# システムプロンプト配置（最優先）
if system_message:
    final_messages.append({"role": "system", "content": system_message})

# 履歴メッセージ（重複防止）
if messages:
    try:
        history_messages = json.loads(messages)
        for msg in history_messages:
            if msg.get("role") == "system" and system_message:
                continue  # 重複防止
            final_messages.append(msg)
    except json.JSONDecodeError:
        log_error("Invalid messages JSON format")

# ユーザーメッセージとビジョンコンテンツの統合
combined_user_content = []
if user_message:
    combined_user_content.append({"type": "text", "text": user_message})
elif prompt:
    combined_user_content.append({"type": "text", "text": prompt})

# 画像がある場合はビジョンコンテンツを追加
if images is not None:
    vision_content, img_metadata_list = build_vision_content(...)
    if isinstance(vision_content, list):
        combined_user_content.extend(vision_content)
    else:
        combined_user_content.append({"type": "image_url", "image_url": {"url": vision_content}})

# 統合されたユーザーメッセージを追加
if combined_user_content:
    final_messages.append({"role": "user", "content": combined_user_content})

# アシスタントメッセージ
if assistant_message:
    final_messages.append({"role": "assistant", "content": assistant_message})

# 安全装置：システムメッセージが先頭にあることを保証
if not final_messages or final_messages[0].get("role") != "system":
    log_debug("Safety net: Inserting empty system prompt as fallback")
    final_messages.insert(0, {"role": "system", "content": ""})
```

#### 2. 詳細なデバッグログの追加 (`llamacpp_client_node.py`)
- `process_request` メソッドに包括的なデバッグログを追加
- 全ての入力パラメータの値を印刷するコードを追加
- システムメッセージの位置検証ロジックを追加

```python
# システムメッセージの位置検証
if final_messages and final_messages[0].get("role") == "system":
    log_debug("✓ System message is correctly positioned at index 0")
else:
    log_error("✗ System message is NOT at index 0! This will cause API errors.")
```

#### 3. メタデータ抽出機能の強化 (`utils/image_utils.py`)
- `build_vision_content` 関数に `extract_metadata` パラメータを追加
- 画像メタデータの抽出機能を強化
- `extract_tensor_metadata` 関数に `batch_index` パラメータを追加

#### 4. 型ヒントの追加 (`utils/llama_client.py`)
- 型ヒントのインポートを整理
- 明示的な型定義を追加

### 技術的詳細

#### メッセージ構築の優先順位
1. **システムプロンプト**: 常に先頭に配置（API 要件）
2. **履歴メッセージ**: 既存の会話履歴（システムプロンプトの重複を防止）
3. **ユーザーメッセージ**: テキストとビジョンコンテンツを統合
4. **アシスタントメッセージ**: 会話の継続

#### 安全装置
- システムメッセージが先頭にあることを保証するためのフォールバック
- 不正な JSON 形式に対するエラーハンドリング
- 画像処理エラーのキャッチとログ出力

### 影響範囲
- **変更されたファイル**: `llamacpp_client_node.py`, `utils/image_utils.py`, `utils/llama_client.py`
- **影響を受ける機能**:
  - `chat_completions` エンドポイントのメッセージ構築ロジック
  - マルチモーダル機能（画像入力）
  - デバッグログ出力

### 検証方法

1. ComfyUI を起動
2. `LlamaCpp Client (Multimodal)` ノードを追加
3. `chat_completions` エンドポイントを選択
4. システムメッセージとユーザーメッセージを設定
5. 画像を入力してマルチモーダル機能が動作するか確認
6. デバッグログでシステムメッセージが先頭に配置されていることを確認

### 関連する変更

- `llamacpp_client_node.py`: メッセージ構築ロジックの完全リファクタリング、デバッグログの追加
- `utils/image_utils.py`: メタデータ抽出機能の強化
- `utils/llama_client.py`: 型ヒントの整理

---

## 2026-03-21: デバッグ機能の追加とパラメータ処理の改善

### 変更内容

#### 1. デバッグログの追加 (`llamacpp_client_node.py`)
- `process_request` メソッドにデバッグ用のログ出力を追加
- 入力パラメータ（server_url, endpoint, prompt）の値を印刷するコードを追加

#### 2. メッセージパースロジックの改善 (`utils/llama_client.py`)
- `handle_chat_completions` メソッド内の `messages` パースロジックを改善
- 従来の try-except パターンから、より堅牢な null チェックベースの処理に変更

#### 3. パラメータクリーンアップの改善 (`utils/param_utils.py`)
- `clean_params` 関数に文字列値のフォールバック処理を追加
- JSON パースできない文字列をそのまま保持するロジックを追加

### 影響範囲
- 全てのエンドポイントでデバッグ情報が出力されるようになります
- パラメータ処理の堅牢性が向上します

---

## 2026-03-21: 実行ロジックの復元とマルチモーダル機能の統合

### 問題の概要

`llamacpp_client_node.py` の `process_request` メソッドで、存在しない `client.execute()` メソッドを呼び出していたため、ノードが機能停止（沈黙）していました。

### 修正内容

#### 1. ロジックの一本化

**問題**: `payload` を構築するブロック（553 行目〜694 行目）と、`kwargs` を構築して実行するブロック（695 行目〜）が重複しており、前者が死んでいました。

**修正**:
- `payload` 構築ブロックを完全に削除
- `kwargs` 構築ブロックにロジックを一本化

#### 2. chat_completions の実行部の修正

**問題**: `messages` パラメータがそのまま kwargs に渡され、`handle_chat_completions` 内でパースされるが、画像処理との連携が不十分でした。

**修正**:
```python
# 修正前
kwargs = {
    "messages": messages,  # 生文字列のまま
    ...
}

# 修正後
final_messages = json.loads(messages) if messages else []
if system_message: final_messages.insert(0, {"role": "system", "content": system_message})
if user_message: final_messages.append({"role": "user", "content": user_message})

# 画像がある場合は vision_content を user message として追加
if images is not None:
    try:
        user_text = prompt or ""
        vision_content, img_metadata_list = build_vision_content(
            user_text=user_text,
            image_data=[],
            tensor_images=images,
            jpeg_quality=DEFAULT_JPEG_QUALITY,
            extract_metadata=extract_metadata,
        )
        final_messages.append({"role": "user", "content": vision_content})
    except Exception as e:
        log_error(f"Error processing images: {e}", e)
        return "", "", str(e), 500, metadata

kwargs = {
    "messages": final_messages,  # 正しく構築されたメッセージリスト
    ...
}
```

#### 3. JSON パースの徹底

**問題**: 一部の JSON 文字列フィールドが json.loads() されずに生文字列のまま渡されていました。

**修正**:
- **infill エンドポイント**:
  - `stop_sequences` → `json.loads(stop_sequences)`
  - `samplers` → `json.loads(samplers)`
  - `logit_bias` → `json.loads(logit_bias)`
  - `lora` → `json.loads(lora)`
  - `input_extra` → `json.loads(input_extra)`
- **reranking エンドポイント**:
  - `documents` → `json.loads(documents)`

#### 4. metadata_list のマージ

**問題**: `metadata_list` が収集されているが、最終的な `metadata` 辞書へのマージが不完全でした。

**修正**:
```python
# 修正後
if metadata_list:
    for i, meta in enumerate(metadata_list):
        metadata[f"image_{i}"] = meta
```

### 3 つの「地雷」の回避

1. **戻り値のアンパック数の不一致**
   - `handle_chat_completions` は 5 つの値を返すため、特別な条件分岐でアンパック
   ```python
   response, raw_response, error, status_code, metadata_list = client.handle_chat_completions(**kwargs)
   ```

2. **引数順序の完全一致**
   - `INPUT_TYPES` で定義された引数順序を `process_request` でも厳密に維持

3. **エラーハンドリング**
   - エラーが発生した場合は `log_error()` でログ出力し、エラーメッセージを返す

### 影響範囲

- **変更されたファイル**: `llamacpp_client_node.py`
- **影響を受ける機能**:
  - 全てのエンドポイント（completion, chat_completions, embeddings, tokenize, detokenize, apply_template, infill, reranking）
  - マルチモーダル機能（画像入力）

### 検証方法

1. ComfyUI を起動
2. `LlamaCpp Client (Multimodal)` ノードを追加
3. 各エンドポイントで正常に動作するか確認
4. 画像を入力してマルチモーダル機能が動作するか確認

### 将来の修正のためのヒント

- `handle_chat_completions` は 5 つの値を返すため、他のエンドポイントとは異なるアンパックが必要
- マルチモーダル機能を使用する場合は、必ず `images` パラメータが None でないことを確認
- JSON パースが必要なフィールドは、すべて `json.loads()` でパースすること