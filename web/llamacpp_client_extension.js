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

function hideWidget(widget) {
    if (!widget) return;

    // Only record original values once and mark hidden
    if (!widget._llamaHidden) {
        widget._llamaOrigType = widget.type;
        // store computeSize if it exists, otherwise null
        widget._llamaOrigComputeSize = typeof widget.computeSize === "function" ? widget.computeSize : null;
        widget._llamaHidden = true;
    }

    // Mark as hidden in a ComfyUI-friendly way
    try {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    } catch (e) {
        console.warn("[LlamaCppClient] hideWidget failed:", e);
    }

    // Hide DOM elements if present
    try {
        if (widget.inputEl) {
            widget.inputEl.style.display = "none";
            widget.inputEl.hidden = true;
            if (widget.inputEl.parentNode && typeof widget.inputEl.parentNode.className === "string" && widget.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                widget.inputEl.parentNode.style.display = "none";
            }
        }
        if (widget.element) {
            widget.element.style.display = "none";
            widget.element.hidden = true;
        }
    } catch (e) {
        console.warn("[LlamaCppClient] hideWidget DOM error:", e);
    }
}

function showWidget(widget) {
    if (!widget) return;
    // Restore type and computeSize if saved
    try {
        if (widget.origType) {
            widget.type = widget.origType;
        }
        if (widget.origComputeSize) {
            widget.computeSize = widget.origComputeSize;
            delete widget.origComputeSize;
        }
    } catch (e) {
        console.warn("[LlamaCppClient] showWidget failed:", e);
    }

    // Show DOM elements if present
    try {
        if (widget.inputEl) {
            widget.inputEl.style.display = "";
            widget.inputEl.hidden = false;
            if (widget.inputEl.parentNode && typeof widget.inputEl.parentNode.className === "string" && widget.inputEl.parentNode.className.indexOf("comfy-multiline") !== -1) {
                widget.inputEl.parentNode.style.display = "block";
            }
        }
        if (widget.element) {
            widget.element.style.display = "";
            widget.element.hidden = false;
        }
    } catch (e) {
        console.warn("[LlamaCppClient] showWidget DOM error:", e);
    }
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

        // 各ウィジェットに対して、表示すべきかどうかを判定して hide/show を適用
        if (!node.widgets) node.widgets = node.masterWidgets ? Array.from(node.masterWidgets) : [];

        for (const w of node.widgets) {
            if (!w) continue;
            if (!w.origType) w.origType = w.type;
            if (showSet.has(w.name)) {
                showWidget(w);
            } else {
                hideWidget(w);
            }
        }

        // ノードサイズの再計算と描画の強制（強化版）
        requestAnimationFrame(function() {
            try {
                // 安全なサイズ再計算（node.size を null にしない）
                if (node._lastComputedSize) {
                    delete node._lastComputedSize;
                }

                if (node.computeSize) {
                    const oldSize = Array.isArray(node.size) ? node.size : [0, 0];
                    const sz = node.computeSize();

                    const MIN_WIDTH = 400;
                    // 最小幅を保証（以前のサイズがあればそれを下限としつつ、十分な幅を確保）
                    sz[0] = Math.max((sz && typeof sz[0] === "number") ? sz[0] : 0, MIN_WIDTH, (oldSize && typeof oldSize[0] === "number") ? oldSize[0] : 0);

                    node.setSize(sz);

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

    // Known numeric constraints to detect obvious corruption from index-shifted values
    const constraints = {
        "dry_base": { min: 1.0 },
        "mirostat_tau": { min: 0.1 },
        "mirostat_eta": { min: 0.001 },
        "min_p": { min: 0.0, max: 1.0 },
        "top_k": { min: 0, max: 10000 },
        "dry_allowed_length": { min: 1 },
        "xtc_threshold": { min: 0.0, max: 1.0 },
        "xtc_probability": { min: 0.0, max: 1.0 }
    };

    function parseNumeric(value, isInt = false) {
        if (typeof value === "number") return value;
        if (typeof value === "string") {
            // try JSON parse first (in case value is "40" or "40.0" or "null")
            try {
                const parsed = JSON.parse(value);
                if (typeof parsed === "number") return isInt ? Math.trunc(parsed) : parsed;
            } catch (e) {
                // JSON.parse failed, fall back
            }
            const n = isInt ? parseInt(value, 10) : parseFloat(value);
            if (!Number.isNaN(n)) return n;
        }
        return NaN;
    }

    for (let i = 0; i < node.widgets.length; i++) {
        const w = node.widgets[i];
        if (!w) continue;

        // 1) Initialize missing or empty values with defaults
        if (w.name in widgetDefaults) {
            const defaultVal = widgetDefaults[w.name];
            if (w.value === null || w.value === undefined || w.value === "" || w.value === "[]") {
                w.value = defaultVal;
            }
        }

        // 2) Ensure JSON parameters remain strings
        if (["stop_sequences", "logit_bias", "samplers", "messages", "tools", "response_format",
             "input_extra", "documents", "lora", "response_fields", "image_data", "dry_sequence_breakers", "tokens"].includes(w.name)) {
            if (!w.value || w.value === "[]") {
                w.value = "[]";
            } else if (typeof w.value !== "string") {
                try {
                    w.value = JSON.stringify(w.value);
                } catch (e) {
                    w.value = "[]";
                }
            }
            continue;
        }

        // 3) Type-correct numeric values and validate ranges for known constrained widgets
        if (w.name in widgetDefaults && typeof widgetDefaults[w.name] === "number") {
            const isInt = Number.isInteger(widgetDefaults[w.name]);
            const parsed = parseNumeric(w.value, isInt);
            if (!Number.isNaN(parsed)) {
                // apply constraints if present
                const c = constraints[w.name];
                if (c) {
                    if (typeof c.min === "number" && parsed < c.min) {
                        console.warn(`[LlamaCppClient] widget ${w.name} value ${parsed} < min ${c.min}, resetting to default`);
                        w.value = widgetDefaults[w.name];
                        continue;
                    }
                    if (typeof c.max === "number" && parsed > c.max) {
                        console.warn(`[LlamaCppClient] widget ${w.name} value ${parsed} > max ${c.max}, resetting to default`);
                        w.value = widgetDefaults[w.name];
                        continue;
                    }
                }
                // if passes checks, store typed value
                w.value = isInt ? Math.trunc(parsed) : parsed;
            } else {
                // parsing failed: likely index-shifted string value — reset to default
                console.warn(`[LlamaCppClient] widget ${w.name} parsing failed for value=${w.value}, resetting to default`);
                w.value = widgetDefaults[w.name];
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
            // Preserve original onSerialize and add name-based serialization
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                // call original serializer if present
                if (onSerialize) {
                    try { onSerialize.apply(this, arguments); } catch (e) { console.warn("[LlamaCppClient] original onSerialize failed:", e); }
                }

                // ensure object exists
                if (!o) o = {};

                try {
                    o._llamaWidgetValues = {};
                    if (this.widgets) {
                        for (const w of this.widgets) {
                            if (w && w.name) {
                                o._llamaWidgetValues[w.name] = w.value;
                            }
                        }
                    }
                } catch (e) {
                    console.warn("[LlamaCppClient] onSerialize error:", e);
                }

                return o;
            };

            // Enhanced onConfigure: restore by name if name-map exists, fallback to defaults via setupNode
            nodeType.prototype.onConfigure = function (info) {
                if (onConfigure) {
                    try { onConfigure.apply(this, arguments); } catch (e) { console.warn("[LlamaCppClient] original onConfigure failed:", e); }
                }

                // If serialized name->value map exists, restore values by name to avoid index mismatch
                try {
                    if (info && info._llamaWidgetValues && this.widgets) {
                        for (const w of this.widgets) {
                            if (w && w.name && (w.name in info._llamaWidgetValues)) {
                                w.value = info._llamaWidgetValues[w.name];
                            }
                        }
                    }
                } catch (e) {
                    console.warn("[LlamaCppClient] name-based restore failed:", e);
                }

                const that = this;
                requestAnimationFrame(function() {
                    setupNode(that);
                });
            };
        }
    }
});
