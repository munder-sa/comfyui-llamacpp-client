console.log("[LlamaCppClient] Extension script loading started...");

import { app } from "../../scripts/app.js";

console.log("[LlamaCppClient] App imported successfully.");

// 共通パラメータのリスト（常に画面に表示される）
const commonParams = [
    "temperature", "top_k", "top_p", "min_p", "seed",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "mirostat", "mirostat_tau", "mirostat_eta", "typical_p",
    "n_keep", "stop_sequences", "ignore_eos", "stream", "n_probs",
    "min_keep", "post_sampling_probs", "return_tokens", "timings_per_token",
    "dynatemp_range", "dynatemp_exponent", "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers", "grammar", "logit_bias", "cache_prompt",
    "id_slot", "samplers", "t_max_predict_ms", "lora"
];

// 各エンドポイントで使用されるパラメータの完全なリスト
// Python の INPUT_TYPES に基づいて完全な定義を行う
const endpointFields = {
    "completion": [
        "prompt", "n_predict", "temperature", "top_k", "top_p", "min_p", "seed",
        "dynatemp_range", "dynatemp_exponent", "xtc_probability", "xtc_threshold",
        "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
        "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
        "dry_sequence_breakers", "mirostat", "mirostat_tau", "mirostat_eta",
        "typical_p", "n_keep", "stop_sequences", "ignore_eos", "stream", "n_probs",
        "min_keep", "post_sampling_probs", "return_tokens", "timings_per_token",
        "grammar", "json_schema", "logit_bias", "cache_prompt", "id_slot", "samplers",
        "t_max_predict_ms", "lora", "response_fields", "image_data"
    ],
    "chat_completions": [
        "messages", "system_message", "user_message", "assistant_message", "max_tokens",
        "model", "tools", "tool_choice", "response_format", "image_data",
        "temperature", "top_k", "top_p", "min_p", "seed", "stream", "stop_sequences",
        "presence_penalty", "frequency_penalty", "n_probs", "min_keep",
        "post_sampling_probs", "return_tokens", "timings_per_token",
        "dynatemp_range", "dynatemp_exponent", "xtc_probability", "xtc_threshold",
        "repeat_penalty", "repeat_last_n", "mirostat", "mirostat_tau", "mirostat_eta",
        "typical_p", "lora"
    ],
    "embeddings": ["input_text", "encoding_format", "embd_normalize", "model"],
    "tokenize": ["content", "add_special", "parse_special", "with_pieces"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": [
        "input_prefix", "input_suffix", "input_extra", "prompt",
        "n_predict", "temperature", "top_k", "top_p", "min_p", "seed",
        "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
        "stop_sequences", "stream", "cache_prompt", "id_slot", "samplers",
        "t_max_predict_ms", "grammar", "logit_bias", "n_probs", "min_keep",
        "post_sampling_probs", "return_tokens", "timings_per_token", "ignore_eos",
        "n_keep", "dynatemp_range", "dynatemp_exponent", "xtc_probability", "xtc_threshold",
        "mirostat", "mirostat_tau", "mirostat_eta", "typical_p", "lora"
    ],
    "reranking": ["query", "documents", "top_n", "model"]
};

// 全てのエンドポイント専用フィールドの集合を作成
const allToggleFieldsMap = {};
for (const key in endpointFields) {
    if (Object.prototype.hasOwnProperty.call(endpointFields, key)) {
        const fields = endpointFields[key];
        for (let i = 0; i < fields.length; i++) {
            allToggleFieldsMap[fields[i]] = true;
        }
    }
}
const allToggleFields = Object.keys(allToggleFieldsMap);

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

        const newWidgets = [];
        for (let i = 0; i < node.masterWidgets.length; i++) {
            const w = node.masterWidgets[i];
            const isToggleField = allToggleFields.indexOf(w.name) !== -1;
            const isCommonParam = commonParams.indexOf(w.name) !== -1;

            let isVisible = true;
            if (isToggleField) {
                // 共通パラメータは常に表示
                if (isCommonParam) {
                    isVisible = true;
                } else {
                    // エンドポイント固有パラメータはエンドポイントに応じて表示
                    isVisible = fieldsToShow.indexOf(w.name) !== -1;
                }
                // images ピンが接続されている場合、image_data を非表示
                if (w.name === "image_data" && isVisible && hasImageLink) {
                    isVisible = false;
                }
            }

            if (isVisible) {
                newWidgets.push(w);
                // DOM 要素（textarea など）を再表示
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
            } else {
                // DOM 要素（textarea など）を隠す
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
