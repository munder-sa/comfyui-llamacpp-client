console.log("[LlamaCppClient] Extension script loading started...");

import { app } from "../../scripts/app.js";

console.log("[LlamaCppClient] App imported successfully.");

// 共通パラメータのリスト（常に画面に表示される）
// 順序：温度制御 → 確率制御 → 繰り返し制御 → ドメイン制御 → ミロスタット制御 → 通常制御 → 時間制御 → その他
const commonParams = [
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
];

// 各エンドポイント固有のパラメータリスト（共通パラメータを除く）
// 順序：エンドポイント固有パラメータ → 共通パラメータ（重複除く）
const endpointSpecificFields = {
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
};

// 特殊パラメータ（最後に配置）
const specialParams = [
    "api_key", "timeout", "image_data", "images", "extract_metadata", "debug_mode"
];

// 共通パラメータのセット（高速検索用）
const commonParamsSet = new Set(commonParams);

// 全てのパラメータ（共通 + 固有）の完全リストをエンドポイントごとに構築
// 順序：共通パラメータ → エンドポイント固有パラメータ（Python 側と一致）
const endpointFields = {};
for (const endpoint in endpointSpecificFields) {
    const specificFields = endpointSpecificFields[endpoint];
    // 共通パラメータ + 固有パラメータ（重複を除く）
    const allFields = [...commonParams, ...specificFields];
    endpointFields[endpoint] = allFields;
}

// 固有パラメータのセット（共通パラメータではないもの）
const allSpecificFields = new Set();
for (const fields of Object.values(endpointSpecificFields)) {
    for (const field of fields) {
        allSpecificFields.add(field);
    }
}

function updateUI(node) {
    try {
        if (!node.masterWidgets) {
            return;
        }

        let endpointWidget = null;
        for (let i = 0; i < node.masterWidgets.length; i++) {
            if (node.masterWidgets[i].name === "endpoint") {
                endpointWidget = node.masterWidgets[i];
                break;
            }
        }

        if (!endpointWidget) {
            return;
        }

        const currentEndpoint = endpointWidget.value;
        const fieldsToShow = endpointFields[currentEndpoint] || [];

        // images ピンの接続チェック - より堅牢な方法でチェック
        let hasImageLink = false;

        // 方法 1: node.inputs をチェック
        if (node.inputs) {
            for (let j = 0; j < node.inputs.length; j++) {
                const input = node.inputs[j];
                if (input && (input.name === "images" || input.type === "IMAGE") && input.link != null) {
                    hasImageLink = true;
                    break;
                }
            }
        }

        // 方法 2: node._ins (入力スロット) をチェック - ComfyUI の内部構造
        if (!hasImageLink && node._ins) {
            for (const slotName in node._ins) {
                const slot = node._ins[slotName];
                if (slot && slot.links && slot.links.length > 0) {
                    hasImageLink = true;
                    break;
                }
            }
        }

        // 方法 3: node.inputs を再確認（link プロパティが配列の場合）
        if (!hasImageLink && node.inputs) {
            for (let j = 0; j < node.inputs.length; j++) {
                const input = node.inputs[j];
                if (input && (input.name === "images" || input.type === "IMAGE")) {
                    // link が配列の場合（新しい ComfyUI 形式）
                    if (Array.isArray(input.link) && input.link.length > 0) {
                        hasImageLink = true;
                        break;
                    }
                    // link が単一値の場合（古い形式）
                    if (input.link != null) {
                        hasImageLink = true;
                        break;
                    }
                }
            }
        }

        // images_data は接続時に非表示にする
        const shouldHideImageData = hasImageLink;

        // 新しいウィジェット配列を構築（順序を保証）
        const newWidgets = [];
        const addedWidgetNames = new Set();

        // 1. 共通パラメータを先に追加（順序を保つ）
        for (const paramName of commonParams) {
            // image_data は接続時にスキップ
            if (paramName === "image_data" && shouldHideImageData) {
                continue;
            }

            // マスターウィジェットから対応するウィジェットを探す
            for (const w of node.masterWidgets) {
                if (w.name === paramName && !addedWidgetNames.has(w.name)) {
                    newWidgets.push(w);
                    addedWidgetNames.add(w.name);

                    // DOM 要素を再表示
                    if (w.inputEl) {
                        w.inputEl.style.display = "block";
                        w.inputEl.hidden = false;
                        if (w.inputEl.parentNode && typeof w.inputEl.parentNode.className === "string" && w.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                            w.inputEl.parentNode.style.display = "block";
                        }
                    }
                    if (w.element) {
                        w.element.style.display = "block";
                        w.element.hidden = false;
                    }
                    break;
                }
            }
        }

        // 2. エンドポイント固有パラメータを追加（順序を保つ）
        const specificFields = endpointSpecificFields[currentEndpoint] || [];
        for (const paramName of specificFields) {
            // image_data は接続時にスキップ
            if (paramName === "image_data" && shouldHideImageData) {
                continue;
            }

            // マスターウィジェットから対応するウィジェットを探す
            for (const w of node.masterWidgets) {
                if (w.name === paramName && !addedWidgetNames.has(w.name)) {
                    newWidgets.push(w);
                    addedWidgetNames.add(w.name);

                    // DOM 要素を再表示
                    if (w.inputEl) {
                        w.inputEl.style.display = "block";
                        w.inputEl.hidden = false;
                        if (w.inputEl.parentNode && typeof w.inputEl.parentNode.className === "string" && w.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                            w.inputEl.parentNode.style.display = "block";
                        }
                    }
                    if (w.element) {
                        w.element.style.display = "block";
                        w.element.hidden = false;
                    }
                    break;
                }
            }
        }

        // 3. 非表示にするウィジェットの DOM を隠す
        for (const w of node.masterWidgets) {
            if (!addedWidgetNames.has(w.name)) {
                // DOM 要素を隠す
                if (w.inputEl) {
                    w.inputEl.style.display = "none";
                    w.inputEl.hidden = true;
                    if (w.inputEl.parentNode && typeof w.inputEl.parentNode.className === "string" && w.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                        w.inputEl.parentNode.style.display = "none";
                    }
                }
                if (w.element) {
                    w.element.style.display = "none";
                    w.element.hidden = true;
                }
            }
        }

        // 実際にウィジェット配列を更新
        node.widgets = newWidgets;

        // ノードサイズの再計算と描画の強制
        requestAnimationFrame(function() {
            if (node.computeSize) {
                const sz = node.computeSize();
                if (sz[0] < node.size[0]) {
                    sz[0] = node.size[0];
                }
                node.setSize(sz);
                if (node.onResize) {
                    node.onResize(sz);
                }
                if (app.graph) {
                    app.graph.setDirtyCanvas(true, true);
                }
            }
        });

    } catch (e) {
        // Error handling silently
    }
}

function initializeWidgetValues(node) {
    if (!node.widgets) return;

    // Create a map of widget names and their default values from the node definition
    const widgetDefaults = {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": -1,
        "max_tokens": -1,
        "n_predict": -1,
        "repeat_penalty": 1.1,
        "timeout": 600,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "mirostat_eta": 0.1,
        "xtc_probability": 0.0,
        "xtc_threshold": 0.1,
        "n_keep": 0,
        "dynatemp_range": 0.0,
        "dynatemp_exponent": 1.0,
        "typical_p": 1.0,
        "dry_multiplier": 0.0,
        "dry_penalty_last_n": -1,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "cache_prompt": true,
        "id_slot": -1,
        "t_max_predict_ms": 0
    };

    for (let i = 0; i < node.widgets.length; i++) {
        const w = node.widgets[i];
        // Initialize widgets with properly typed values
        if (w.name in widgetDefaults) {
            const defaultVal = widgetDefaults[w.name];
            if (w.value === null || w.value === undefined || w.value === "" || w.value === "[]") {
                w.value = defaultVal;
            }
        }
        // Ensure JSON parameters are strings
        if (["stop_sequences", "logit_bias", "samplers", "messages", "tools", "response_format",
             "input_extra", "documents", "lora", "response_fields", "image_data", "dry_sequence_breakers", "tokens"].includes(w.name)) {
            if (!w.value || w.value === "[]") {
                w.value = "[]";
            } else if (typeof w.value !== "string") {
                w.value = JSON.stringify(w.value);
            }
        }
    }
}

function setupNode(node) {
    if (!node.widgets) {
        return;
    }

    // マスターのウィジェットリストを保存しておく (Convert to Input で減る場合も考慮)
    if (!node.masterWidgets) {
        node.masterWidgets = node.widgets ? Array.from(node.widgets) : [];
    }

    // Initialize widget values to safe defaults
    initializeWidgetValues(node);

    let endpointWidget = null;
    for (let i = 0; i < node.masterWidgets.length; i++) {
        if (node.masterWidgets[i].name === "endpoint") {
            endpointWidget = node.masterWidgets[i];
            break;
        }
    }

    if (endpointWidget) {
        if (!endpointWidget._llamaCbWrapped) {
            const origCallback = endpointWidget.callback;
            endpointWidget.callback = function() {
                if (origCallback) {
                    origCallback.apply(this, arguments);
                }
                updateUI(node);
            };
            endpointWidget._llamaCbWrapped = true;
        }
    }

    // 接続状態の変更を監視 (images ピン用)
    if (!node._llamaConnWrapped) {
        const onConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }

            // type === 1 (LiteGraph.INPUT)
            if (type === 1) {
                let isImagePin = false;
                if (this.inputs && this.inputs[index]) {
                    if (this.inputs[index].name === "images" || this.inputs[index].type === "IMAGE") {
                        isImagePin = true;
                    }
                }

                if (isImagePin) {
                    const that = this;
                    setTimeout(function() {
                        updateUI(that);
                    }, 200);
                }
            }
        };
        node._llamaConnWrapped = true;
    }

    // 初回の UI 更新
    updateUI(node);
}

app.registerExtension({
    name: "LlamaCppClient.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // ノード名の一致チェック（LlamaCppClientNode または LlamaCppClient）
        if (nodeData.name === "LlamaCppClientNode" || nodeData.name === "LlamaCppClient") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                const that = this;
                requestAnimationFrame(function() {
                    setupNode(that);
                });
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }

                const that = this;
                requestAnimationFrame(function() {
                    setupNode(that);
                });
            };
        }
    }
});
