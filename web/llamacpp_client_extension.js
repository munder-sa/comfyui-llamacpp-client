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

// 各エンドポイントの「優先パラメータ」（最上部に表示するもの）
// 順序：優先パラメータ → 共通パラメータ → その他固有パラメータ → 特殊パラメータ
const endpointPriorityFields = {
    "completion": ["prompt"],
    "chat_completions": ["system_message", "user_message"],
    "embeddings": ["input_text"],
    "tokenize": ["content"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": ["input_prefix", "input_suffix"],
    "reranking": ["query", "documents"]
};

// 優先パラメータのセット（共通パラメータと重複しないようにチェック用）
const priorityFieldsSet = new Set();
for (const fields of Object.values(endpointPriorityFields)) {
    for (const field of fields) {
        priorityFieldsSet.add(field);
    }
}

// 優先パラメータ以外の固有パラメータ（共通パラメータと重複除く）
const endpointOtherSpecificFields = {};
for (const endpoint in endpointSpecificFields) {
    const allSpecific = endpointSpecificFields[endpoint];
    const priority = endpointPriorityFields[endpoint] || [];
    const prioritySet = new Set(priority);
    // 優先パラメータと共通パラメータを除いたものを「その他固有」として残す
    const otherSpecific = allSpecific.filter(f => !prioritySet.has(f) && !commonParamsSet.has(f));
    endpointOtherSpecificFields[endpoint] = otherSpecific;
}

// 全てのパラメータ（優先 + 共通 + その他固有）の完全リストをエンドポイントごとに構築
// 順序：優先パラメータ → 共通パラメータ → その他固有パラメータ → 特殊パラメータ
const endpointFields = {};
for (const endpoint in endpointSpecificFields) {
    const priorityFields = endpointPriorityFields[endpoint] || [];
    const otherSpecificFields = endpointOtherSpecificFields[endpoint] || [];
    // 重複を除きながら順序を保つ
    const seen = new Set();
    const allFields = [];

    // 1. 優先パラメータ
    for (const f of priorityFields) {
        if (!seen.has(f)) {
            allFields.push(f);
            seen.add(f);
        }
    }

    // 2. 共通パラメータ
    for (const f of commonParams) {
        if (!seen.has(f)) {
            allFields.push(f);
            seen.add(f);
        }
    }

    // 3. その他固有パラメータ
    for (const f of otherSpecificFields) {
        if (!seen.has(f)) {
            allFields.push(f);
            seen.add(f);
        }
    }

    endpointFields[endpoint] = allFields;
}

function isConvertedToInput(node, widgetName) {
    if (!node) return false;

    // Check exposed inputs array (standard ComfyUI)
    if (node.inputs && Array.isArray(node.inputs)) {
        for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            if (!input) continue;
            if (input.name === widgetName) {
                return true;
            }
        }
    }

    // Check internal _ins mapping (ComfyUI internal representation)
    if (node._ins) {
        for (const key in node._ins) {
            const slot = node._ins[key];
            if (!slot) continue;
            if (slot.name === widgetName) return true;
        }
    }

    return false;
}

function updateUI(node) {
    try {
        if (!node.masterWidgets || !node.widgets) {
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

        // 表示するパラメータのセットを作成
        const showSet = new Set();

        // 優先パラメータ
        const priorityFields = endpointPriorityFields[currentEndpoint] || [];
        for (const f of priorityFields) {
            if (f !== "image_data" || !shouldHideImageData) {
                showSet.add(f);
            }
        }

        // 共通パラメータ
        for (const f of commonParams) {
            if (f !== "image_data" || !shouldHideImageData) {
                showSet.add(f);
            }
        }

        // その他固有パラメータ
        const otherSpecificFields = endpointOtherSpecificFields[currentEndpoint] || [];
        for (const f of otherSpecificFields) {
            if (f !== "image_data" || !shouldHideImageData) {
                showSet.add(f);
            }
        }

        // 特殊パラメータ
        for (const f of specialParams) {
            if (f !== "image_data" || !shouldHideImageData) {
                showSet.add(f);
            }
        }

        // endpoint と server_url ウィジェットは常に含める（requiredパラメータ）
        showSet.add("endpoint");
        showSet.add("server_url");

        // 表示対象のウィジェットのみを順序通りに構築
        const newWidgetOrder = [];
        const addedNames = new Set();

        // 1. Requiredパラメータ（endpoint, server_url）を先頭に
        const requiredParams = ["endpoint", "server_url"];
        for (const name of requiredParams) {
            const w = node.masterWidgets.find(mw => mw.name === name);
            if (w && showSet.has(name) && !addedNames.has(w.name)) {
                newWidgetOrder.push(w);
                addedNames.add(w.name);
            }
        }

        // 2. 優先パラメータ
        for (const name of priorityFields) {
            if (name === "image_data" && shouldHideImageData) continue;
            const w = node.masterWidgets.find(mw => mw.name === name);
            if (w && showSet.has(name) && !addedNames.has(w.name)) {
                newWidgetOrder.push(w);
                addedNames.add(w.name);
            }
        }

        // 3. 共通パラメータ
        for (const name of commonParams) {
            if (name === "image_data" && shouldHideImageData) continue;
            const w = node.masterWidgets.find(mw => mw.name === name);
            if (w && showSet.has(name) && !addedNames.has(w.name)) {
                newWidgetOrder.push(w);
                addedNames.add(w.name);
            }
        }

        // 4. その他固有パラメータ
        for (const name of otherSpecificFields) {
            if (name === "image_data" && shouldHideImageData) continue;
            const w = node.masterWidgets.find(mw => mw.name === name);
            if (w && showSet.has(name) && !addedNames.has(w.name)) {
                newWidgetOrder.push(w);
                addedNames.add(w.name);
            }
        }

        // 5. 特殊パラメータ
        for (const name of specialParams) {
            if (name === "image_data" && shouldHideImageData) continue;
            const w = node.masterWidgets.find(mw => mw.name === name);
            if (w && showSet.has(name) && !addedNames.has(w.name)) {
                newWidgetOrder.push(w);
                addedNames.add(w.name);
            }
        }

        // ウィジェット配列を完全に置き換え（表示対象のみ）
        node.widgets = newWidgetOrder;

        // マスターウィジェットのDOM要素を適切に制御
        // 表示対象のウィジェットは表示、非表示対象はDOMから隠す
        for (const masterWidget of node.masterWidgets) {
            if (showSet.has(masterWidget.name)) {
                // 表示対象：DOM要素を表示
                if (masterWidget.inputEl) {
                    masterWidget.inputEl.style.display = "block";
                    masterWidget.inputEl.hidden = false;
                    if (masterWidget.inputEl.parentNode && typeof masterWidget.inputEl.parentNode.className === "string" && masterWidget.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                        masterWidget.inputEl.parentNode.style.display = "block";
                    }
                }
                if (masterWidget.element) {
                    masterWidget.element.style.display = "block";
                    masterWidget.element.hidden = false;
                }
            } else {
                // 非表示対象：DOM要素を完全に隠す
                if (masterWidget.inputEl) {
                    masterWidget.inputEl.style.display = "none";
                    masterWidget.inputEl.hidden = true;
                    if (masterWidget.inputEl.parentNode && typeof masterWidget.inputEl.parentNode.className === "string" && masterWidget.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                        masterWidget.inputEl.parentNode.style.display = "none";
                    }
                }
                if (masterWidget.element) {
                    masterWidget.element.style.display = "none";
                    masterWidget.element.hidden = true;
                }
            }
        }

        // ノードサイズの再計算と描画の強制（強化版）
        requestAnimationFrame(function() {
            try {
                // サイズキャッシュをリセット（ComfyUI内部）
                node.size = null;  // サイズキャッシュをクリア
                if (node._lastComputedSize) {
                    delete node._lastComputedSize;
                }

                // 強制的にサイズ再計算
                if (node.computeSize) {
                    const sz = node.computeSize();

                    // 最小幅を保証
                    if (sz[0] < node.size?.[0]) {
                        sz[0] = node.size[0];
                    }

                    // サイズを確定
                    node.setSize(sz);

                    // リサイズコールバック実行
                    if (node.onResize) {
                        node.onResize(sz);
                    }
                }

                // キャンバスに強制再描画を要求
                if (app.graph) {
                    app.graph.setDirtyCanvas(true, true);
                }
            } catch (e) {
                console.error("[LlamaCppClient] Size recalc error:", e);
            }
        });

    } catch (e) {
        console.error("[LlamaCppClient] updateUI error:", e);
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
