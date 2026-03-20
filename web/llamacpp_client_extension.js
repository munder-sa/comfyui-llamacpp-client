console.log("[LlamaCppClient] Extension script loading started...");

import { app } from "../../scripts/app.js";

console.log("[LlamaCppClient] App imported successfully.");

const endpointFields = {
    "completion": ["prompt"],
    "chat_completions": ["messages", "system_message", "user_message", "assistant_message", "max_tokens", "model", "tools", "tool_choice", "response_format", "image_data"],
    "embeddings": ["input_text", "encoding_format", "embd_normalize", "model"],
    "tokenize": ["content", "add_special", "parse_special", "with_pieces"],
    "detokenize": ["tokens"],
    "apply_template": ["messages"],
    "infill": ["input_prefix", "input_suffix", "input_extra"],
    "reranking": ["query", "documents", "top_n", "model"]
};

// 全てのエンドポイント専用フィールドの集合を作成
let allToggleFieldsMap = {};
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
        if (!node.masterWidgets) return;
        
        let endpointWidget = null;
        for (let i = 0; i < node.masterWidgets.length; i++) {
            if (node.masterWidgets[i].name === "endpoint") {
                endpointWidget = node.masterWidgets[i];
                break;
            }
        }
        
        if (!endpointWidget) return;
        
        const currentEndpoint = endpointWidget.value;
        const fieldsToShow = endpointFields[currentEndpoint] || [];

        // imagesピンの接続チェック
        let hasImageLink = false;
        if (node.inputs) {
            for (let j = 0; j < node.inputs.length; j++) {
                if ((node.inputs[j].name === "images" || node.inputs[j].type === "IMAGE") && node.inputs[j].link != null) {
                    hasImageLink = true;
                    break;
                }
            }
        }

        const newWidgets = [];
        for (let i = 0; i < node.masterWidgets.length; i++) {
            const w = node.masterWidgets[i];
            const isToggleField = allToggleFields.indexOf(w.name) !== -1;
            
            let isVisible = true;
            if (isToggleField) {
                isVisible = fieldsToShow.indexOf(w.name) !== -1;
                if (w.name === "image_data" && isVisible && hasImageLink) {
                    isVisible = false;
                }
            }

            if (isVisible) {
                newWidgets.push(w);
                // DOM要素（textareaなど）を再表示
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
                // DOM要素（textareaなど）を隠す
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
        console.error("[LlamaCppClient] Error in updateUI:", e);
    }
}

function setupNode(node) {
    if (!node.widgets) return;

    // マスターのウィジェットリストを保存しておく (Convert to Inputなどで減る場合も考慮)
    if (!node.masterWidgets) {
        node.masterWidgets = Array.from(node.widgets);
    }

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

    // 初回のUI更新
    updateUI(node);
}

app.registerExtension({
    name: "LlamaCppClient.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LlamaCppClient") {
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
